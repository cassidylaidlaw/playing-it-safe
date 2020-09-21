
from typing import List, Tuple
import torch
import torch.autograd

import attacks


def run_validation_attacks(model, normalizer, val_loader,
                           writer, iteration, args, val_batches=None):
    """
    Runs attacks on the model at the end of each iteration to see
    how well it is performing.
    """

    if val_batches is None:
        val_batches = args.val_batches

    validation_attacks: List[Tuple[attacks.Attack, float]]
    if args.abstain:
        validation_attacks = [
            (attacks.none, 1),  # type: ignore
            (attacks.pgd_abstain_sum_linf, 1),  # type: ignore
            (attacks.pgd_abstain_interp_linf, 1),  # type: ignore
            # only attack 20% of batches with DeepFool since it takes longer
            (attacks.deepfool_linf, 0.2),  # type: ignore
            (attacks.deepfool_abstain_linf, 0.2),  # type: ignore
        ]
    else:
        validation_attacks = [
            (attacks.none, 1),  # type: ignore
            (attacks.pgd_linf, 1),  # type: ignore
            # only attack 20% of batches with DeepFool since it takes longer
            (attacks.deepfool_linf, 0.2),  # type: ignore
        ]

    for attack, batch_proportion in validation_attacks:
        batches_correct = []
        if args.abstain:
            batches_abstain = []
        for batch_index, (inputs, labels) in enumerate(val_loader):
            if batch_index >= val_batches * batch_proportion:
                break

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            logits = model(normalizer(inputs))
            adv_inputs = attack(model, normalizer, inputs, labels,
                                args.dataset, abstain=args.abstain)
            adv_logits = model(normalizer(adv_inputs)).detach()

            batches_correct.append(adv_logits.argmax(1) == labels)
            success = (
                (logits.argmax(1) == labels) &  # was classified correctly
                (adv_logits.argmax(1) != labels)  # and now is not
            )
            if args.abstain:
                abstain_class = adv_logits.size()[1] - 1
                batches_abstain.append(adv_logits.argmax(1) == abstain_class)
                success = success & (adv_logits.argmax(1) != abstain_class)
        print_cols = [f'ATTACK {attack.__name__}']

        correct = torch.cat(batches_correct)
        accuracy = correct.float().mean()
        writer.add_scalar(f'val/{attack.__name__}/accuracy',
                          accuracy.item(),
                          iteration)
        print_cols.append(f'accuracy: {accuracy.item() * 100:.1f}%')

        if args.abstain:
            abstain = torch.cat(batches_abstain)
            abstain_prop = abstain.float().mean()
            writer.add_scalar(f'val/{attack.__name__}/abstain',
                              abstain_prop.item(),
                              iteration)
            print_cols.append(f'abstain: {abstain_prop.item() * 100:.1f}%')

            incorrect_prop = 1 - accuracy - abstain_prop
            writer.add_scalar(f'val/{attack.__name__}/incorrect',
                              incorrect_prop.item(),
                              iteration)
            print_cols.append(f'incorrect: {incorrect_prop.item() * 100:.1f}%')

        print(*print_cols, sep='\t')
