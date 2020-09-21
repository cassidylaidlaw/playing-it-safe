from typing import Iterable, List, Tuple
import torch
import csv
import argparse
from torch.utils.data import DataLoader
from itertools import islice
from collections import Counter

import utils
import attacks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, help='architecture name')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file',
                        required=False)
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, default=-1,
                        help='number of batches to evaluate over')
    parser.add_argument('--abstain', action='store_true', default=False,
                        help='whether to include an abstain class')
    parser.add_argument('--activation', type=str, default='softplus',
                        help='activation function to use')
    parser.add_argument('--out', type=str, required=False,
                        help='output path (without extension)')
    parser.add_argument('attacks', type=str, nargs='*',
                        help='attacks to evaluate against')

    args = parser.parse_args()

    utils.activation.activation = args.activation  # type: ignore

    dataset, model, normalizer = utils.load_dataset_model(args)

    if args.checkpoint:
        state = torch.load(args.checkpoint)
        model.load_state_dict(state['model'])

    model.eval()

    val_loader = DataLoader(dataset(False),
                            shuffle=False,
                            batch_size=args.batch_size)

    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    if args.num_batches > 0:
        batches = islice(val_loader, args.num_batches)
    else:
        batches = val_loader

    def check_label(output_label, correct_label) -> str:
        if output_label == correct_label:
            return 'C'
        elif args.abstain and output_label == abstain_label:
            return 'A'
        else:
            return 'I'

    all_results: List[List[str]] = []

    for batch_index, (inputs, labels) in enumerate(batches):
        print(f'BATCH {batch_index:03d}')
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        orig_logits = model(normalizer(inputs))
        orig_labels = orig_logits.argmax(1)

        attacks_labels = []

        # wrap in function to prevent memory leaks
        def run_attack(attack_name):
            attack = getattr(attacks, attack_name)
            adv_inputs = attack(
                model, normalizer, inputs, labels, args.dataset,
                abstain=args.abstain)
            attacks_labels.append(
                model(normalizer(adv_inputs)).argmax(1).clone().detach())

        for attack_name in args.attacks:
            run_attack(attack_name)

        if args.abstain:
            abstain_label = orig_logits.size()[1] - 1

        for all_labels in \
                zip(labels, orig_labels, *attacks_labels):
            correct_label = all_labels[0]
            classified_labels = all_labels[1:]
            all_results.append([
                check_label(classified_label, correct_label)
                for classified_label in classified_labels
            ])

    if args.out:
        with open(f'{args.out}.csv', 'w') as out_file:
            out_csv = csv.writer(out_file)
            out_csv.writerow(['orig'] + args.attacks)
            for row in all_results:
                out_csv.writerow(row)

    for attack_index, attack_name in enumerate(['none'] + args.attacks):
        attack_results = Counter(row[attack_index] for row in all_results)
        print_columns = [
            f"correct: {attack_results['C'] / len(all_results) * 100:.1f}%",
            f"incorrect: {attack_results['I'] / len(all_results) * 100:.1f}%",
        ]
        if args.abstain:
            print_columns[1:1] = [
                'abstain: '
                f"{attack_results['A'] / len(all_results) * 100:.1f}%",
            ]
        print(f'ATTACK {attack_name}',
              *print_columns,
              sep='\t')

    union_error = sum(
        any(attack_result == 'I' for attack_result in row)
        for row in all_results
    ) / len(all_results)
    print('UNION', f'error = {union_error * 100:.1f}%', sep='\t')
