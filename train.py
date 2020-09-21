from typing import Any, DefaultDict, Dict, List
import torch
import argparse
import os
import math
import shutil
import glob
import time
import builtins
import random
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import defaultdict

import utils
import losses
import validation

LOSS_FUNCTIONS = {
    'l_adv': losses.AdversarialLoss,
    'l_adv_abstain': losses.AdversarialAbstainLoss,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, help='architecture name')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs trained')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--activation', type=str, default='softplus',
                        help='activation function to use')
    parser.add_argument('--val_batches', type=int, default=10,
                        help='number of batches to validate on each epoch')
    parser.add_argument('--log_dir', type=str, default='data/logs')
    parser.add_argument('--abstain', action='store_true', default=False,
                        help='whether to include an abstain class')

    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--continue', default=False, action='store_true',
                        help='continue previous training')
    parser.add_argument('--keep_every', type=int, default=1,
                        help='only keep a checkpoint every X epochs')

    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, metavar='LR', required=False,
                        help='learning rate')
    parser.add_argument('--lr_drop_epoch', type=float, required=False,
                        help='first epoch when the learning rate should drop')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='clip gradients to this value')

    parser.add_argument('--loss', type=str, help='loss function to use')
    parser.add_argument('--base_loss', type=str,
                        choices=['diff', 'sdiff', 'ce'], default='diff',
                        help='the loss to compute the gradient of for '
                        'higher-order losses')
    parser.add_argument('--val_loss', type=str, default='l_ce',
                        help='loss function to compute during validation')

    args = parser.parse_args()

    if args.optim == 'adam':
        if args.lr is None:
            args.lr = 1e-3
        if args.lr_drop_epoch is None:
            args.lr_drop_epoch = 120
    elif args.optim == 'sgd':
        if args.lr is None:
            args.lr = 1e-1
        if args.lr_drop_epoch is None:
            args.lr_drop_epoch = 75

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    utils.calculate_base_loss.base_loss = args.base_loss  # type: ignore
    utils.activation.activation = args.activation  # type: ignore

    dataset, model, normalizer = utils.load_dataset_model(args)
    print('Constructing datasets...')
    train_loader = DataLoader(dataset(True),
                              shuffle=True,
                              batch_size=args.batch_size)
    val_loader = DataLoader(dataset(False),
                            shuffle=True,
                            batch_size=args.batch_size)

    experiment_path_parts = [args.dataset, args.arch]
    if args.base_loss != 'diff':
        experiment_path_parts.append(args.base_loss)
    if args.activation != 'softplus':
        experiment_path_parts.append(args.activation)
    if args.optim != 'sgd':
        experiment_path_parts.append(args.optim)
    experiment_path_parts.append(args.loss)
    experiment_path = os.path.join(*experiment_path_parts)

    optimizer: optim.Optimizer
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=2e-4)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters())
    else:
        raise ValueError(f'invalid optimizer {args.optim}')

    iteration = 0
    log_dir = os.path.join(args.log_dir, experiment_path)
    if os.path.exists(log_dir):
        print(f'The log directory {log_dir} exists, delete? (y/N) ', end='')
        if not vars(args)['continue'] and input().strip() == 'y':
            shutil.rmtree(log_dir)
            # sleep necessary to prevent weird bug where directory isn't
            # actually deleted
            time.sleep(5)
    writer = SummaryWriter(log_dir)

    # check for checkpoints
    def get_checkpoint_fnames():
        for checkpoint_fname in glob.glob(os.path.join(glob.escape(log_dir),
                                                       '*.ckpt.pth')):
            epoch = int(os.path.basename(checkpoint_fname).split('.')[0])
            if epoch < args.num_epochs:
                yield epoch, checkpoint_fname

    start_epoch = 0
    latest_checkpoint_epoch = -1
    latest_checkpoint_fname = None
    for epoch, checkpoint_fname in get_checkpoint_fnames():
        if epoch > latest_checkpoint_epoch:
            latest_checkpoint_epoch = epoch
            latest_checkpoint_fname = checkpoint_fname
    if latest_checkpoint_fname is not None:
        print(f'Load checkpoint {latest_checkpoint_fname}? (Y/n) ', end='')
        if vars(args)['continue'] or input().strip() != 'n':
            state = torch.load(latest_checkpoint_fname)
            iteration = state['iteration']
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = latest_checkpoint_epoch + 1

    # necessary to put training loop in a function because otherwise we get
    # huge memory leaks
    def run_iter(inputs, labels, iteration, train=True, log_fn=None):
        prefix = 'train' if train else 'val'
        loss_eq = args.loss if train else args.val_loss
        if log_fn is None:
            def log_fn(tag, value):
                if isinstance(value, float) or len(value.size()) == 0:
                    writer.add_scalar(f'{prefix}/{tag}', value, iteration)
                elif len(value.size()) == 3 and iteration % 100 == 0:
                    writer.add_image(f'{prefix}/{tag}', value, iteration)
                elif len(value.size()) == 4 and iteration % 100 == 0:
                    writer.add_images(f'{prefix}/{tag}', value, iteration)

        model.eval()  # set model to eval to generate adversarial examples

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        non_batch_dims = inputs.size()[1:]

        inputs_to_compute: List[torch.Tensor] = []
        computed_logits: List[torch.Tensor] = []
        computed_grads: List[torch.Tensor] = []

        loss_objects: Dict[str, losses.Loss] = {}
        for loss_name, Loss in LOSS_FUNCTIONS.items():
            loss_objects[loss_name] = Loss(
                model, normalizer, inputs, labels, args.dataset,
                log_fn, train,
                inputs_to_compute, computed_logits,
            )

        # DETERMINE INPUTS TO RUN MODEL OVER
        inputs_to_compute.append(inputs)

        first_eval_locals: DefaultDict[str, Any] = \
            defaultdict(lambda: math.nan)
        first_eval_locals.update(loss_objects)
        first_eval_locals.update(globals())
        first_eval_locals.update(builtins.__dict__)
        eval(loss_eq, None, first_eval_locals)

        # FORWARD PASS
        if train:
            optimizer.zero_grad()
            model.train()  # now we set the model to train mode

        for inputs in inputs_to_compute:
            inputs.requires_grad = True
        all_inputs = torch.cat(inputs_to_compute)

        all_logits = model(normalizer(all_inputs))

        # CALCULATE GRADIENT
        base_loss = utils.calculate_base_loss(
            all_logits,
            torch.cat([labels] * len(inputs_to_compute)),
        )
        all_grads, = torch.autograd.grad(base_loss.sum(), all_inputs,
                                         create_graph=True,
                                         retain_graph=True)
        for i in range(len(inputs_to_compute)):
            sl = slice(i * len(inputs), (i + 1) * len(inputs))
            computed_logits.append(all_logits[sl])
            computed_grads.append(all_grads[sl])

        # CONSTRUCT LOSS
        logits = computed_logits[0]
        grads = computed_grads[0]

        base_loss = utils.calculate_base_loss(logits, labels)
        grads_magnitude = (grads ** 2) \
            .sum(dim=list(range(1, len(grads.size())))) \
            .sqrt()

        l_base = torch.mean(base_loss)
        l_ce = nn.CrossEntropyLoss()(logits, labels)
        if 'diff' in args.base_loss:
            l_diff = l_base
        l_grad = torch.mean(grads_magnitude)
        l_grad_l1 = grads.abs().sum(
            dim=list(range(1, len(grads.size())))).mean()

        for loss_object in loss_objects.values():
            loss_object.switch_to_eval()
        second_eval_locals = dict(locals())
        second_eval_locals.update(loss_objects)
        second_eval_locals.update(builtins.__dict__)
        loss = eval(loss_eq, None, second_eval_locals)

        # LOGGING
        accuracy = utils.calculate_accuracy(logits, labels)
        log_fn('loss/loss', loss.item())
        log_fn('loss/ce', l_ce.item())
        log_fn('loss/base', l_base.item())
        if 'diff' in args.base_loss:
            log_fn('loss/diff', l_diff.item())
        log_fn('loss/grad', l_grad.item())
        log_fn('loss/grad_l1', l_grad_l1.item())
        log_fn('accuracy', accuracy.item())
        if train:
            print(f'ITER {iteration:06d}',
                  f'accuracy: {accuracy.item() * 100:5.1f}%',
                  f'loss: {loss.item():.2f}',
                  sep='\t')

        if train and iteration % 1000 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'train/model/{name}',
                                     param.clone().cpu().data.numpy(),
                                     iteration)

        # OPTIMIZATION
        if train:
            loss.backward()

            # log gradient norm
            param_grads = torch.cat([param.grad.data.view(-1) for param in
                                     model.parameters()])
            log_fn('param_grad_l2', param_grads.norm(p=2).item())
            log_fn('param_grad_linf', param_grads.norm(p=math.inf).item())

            # clip gradients and optimize
            nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
            optimizer.step()

    for epoch in range(start_epoch, args.num_epochs):
        if args.optim == 'sgd':
            if epoch < args.lr_drop_epoch:
                lr = args.lr
            elif epoch < args.lr_drop_epoch + 15:
                lr = args.lr * 0.1
            elif epoch < args.lr_drop_epoch + 25:
                lr = args.lr * 0.01
            else:
                lr = args.lr * 0.001
        elif args.optim == 'adam':
            if epoch < args.lr_drop_epoch:
                lr = args.lr
            else:
                lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # some models need to know the current epoch
        model.epoch = epoch

        print(f'START EPOCH {epoch:04d} (lr={lr:.0e})')
        for batch_index, (inputs, labels) in enumerate(train_loader):
            if batch_index >= 20:
                break
            if epoch < 10 and args.optim == 'sgd':
                lr = (iteration + 1) / (10 * len(train_loader)) * args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            run_iter(inputs, labels, iteration)
            iteration += 1
        print(f'END EPOCH {epoch:04d}')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # VALIDATION
        print('BEGIN VALIDATION')
        model.eval()

        validation.run_validation_attacks(
            model, normalizer, val_loader, writer, iteration, args)

        print('VAL LOSS')
        val_logs = defaultdict(list)
        for batch_index, (inputs, labels) in enumerate(val_loader):
            run_iter(inputs, labels, iteration, train=False,
                     log_fn=lambda tag, value: val_logs[tag].append(value))
        for tag, values in val_logs.items():
            if isinstance(values[0], float) or (values[0].size()) == 0:
                writer.add_scalar(f'val_loss/{tag}',
                                  torch.mean(torch.Tensor(values)),
                                  iteration)
            else:
                writer.add_histogram(
                    f'val_loss/{tag}',
                    torch.cat(values).clone().cpu().data.numpy(),
                    iteration,
                )

        print('END VALIDATION')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        checkpoint_fname = os.path.join(log_dir, f'{epoch:04d}.ckpt.pth')
        print(f'CHECKPOINT {checkpoint_fname}')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration,
            'model_class': args.arch,
        }
        torch.save(state, checkpoint_fname)

        # delete extraneous checkpoints
        last_keep_checkpoint = (epoch // args.keep_every) * args.keep_every
        for epoch, checkpoint_fname in get_checkpoint_fnames():
            if epoch < last_keep_checkpoint and epoch % args.keep_every != 0:
                print(f'  remove {checkpoint_fname}')
                os.remove(checkpoint_fname)

    print('BEGIN EVALUATION')
    model.eval()
    validation.run_validation_attacks(
        model, normalizer, val_loader,
        SummaryWriter(os.path.join('/tmp', experiment_path)),
        iteration, args,
        val_batches=len(val_loader),
    )
    print('END EVALUATION')
