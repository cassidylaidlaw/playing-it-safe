import torch
import torch.autograd
from torch import nn
import torch.functional as F
import math

import torchvision
from torchvision.datasets import MNIST, CIFAR10, SVHN
from typing import Callable

from models import DifferentiableNormalizer
from models import *  # noqa: F401, F403
from datasets import TwoClassMNIST


def load_dataset_model(args):
    """
    Given argparse arguments args with at least the arguments 'dataset' and
    'arch', returns a tuple (dataset, model, normalizer).
    """

    normalizer: Callable[[torch.Tensor], torch.Tensor]

    if args.dataset in ['mnist', 'mnist2']:
        transform = torchvision.transforms.ToTensor()
        normalizer = DifferentiableNormalizer((0.1307,), (0.3081,))

        def dataset(train):
            return {
                'mnist': MNIST,
                'mnist2': TwoClassMNIST,
            }[args.dataset](
                '~/datasets/mnist',
                train=train,
                download=True,
                transform=transform,
            )
    elif args.dataset == 'svhn':
        transform = torchvision.transforms.ToTensor()
        normalizer = DifferentiableNormalizer(
            (0.507, 0.487, 0.441),
            (0.267, 0.256, 0.276),
        )

        def dataset(train):
            return SVHN(
                '~/datasets/svhn',
                split='train' if train else 'test',
                download=True,
                transform=transform,
            )
    elif args.dataset == 'cifar10':
        normalizer = DifferentiableNormalizer(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )

        def dataset(train):
            if train:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32, 4),
                    torchvision.transforms.ToTensor(),
                ])
            else:
                transform = torchvision.transforms.ToTensor()

            return CIFAR10(
                '~/datasets/cifar10',
                train=train,
                download=True,
                transform=transform,
            )
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}.')

    num_classes = {
        'mnist': 10,
        'mnist2': 2,
        'svhn': 10,
        'cifar10': 10,
    }[args.dataset]
    if getattr(args, 'abstain', False):
        num_classes += 1

    print('Constructing model...')

    model = eval(args.arch)(num_classes=num_classes)
    if torch.cuda.is_available():
        model.cuda()

    if getattr(args, 'no_normalizer', False):
        normalizer = lambda x: x  # noqa: E731

    return dataset, model, normalizer


def ce_loss_with_abstain(
    logits: torch.Tensor,
    labels: torch.Tensor,
    abstain: bool = False,
    abstain_cost: float = 0,
    reduction='mean',
) -> torch.Tensor:
    """
    If abstain is False, calculates the normal cross-entropy loss between
    the logits and the labels. If abstain is True, then multiplies this
    element-wise with the cross-entropy loss between the logits and the
    abstain class (the last one). In this way, the loss function is lower when
    the classifier either outputs the correct class OR the abstain class.
    """

    ce_loss = nn.CrossEntropyLoss(reduction='none')
    loss = ce_loss(logits, labels)
    if abstain:
        abstain_labels = torch.ones_like(labels) * logits.size()[1] - 1
        abstain_loss = ce_loss(logits, abstain_labels)
        abstain_loss = abstain_cost + (1 - abstain_cost) * abstain_loss
        loss *= abstain_loss

    from losses import Loss
    return Loss.reduce(loss, reduction)


def calculate_base_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    abstain: bool = False,
    abstain_cost: float = 0,
) -> torch.Tensor:
    """
    Calculates a loss between the given logits (output from classifier) and
    labels (longs). The loss function used is chosen based on
    calculate_base_loss.base_loss, which can be one of the following:

    'ce': cross-entropy loss.

    'diff':
    The difference between the greatest incorrect logit and the logit
    corresponding to the ground-truth label.

    'sdiff':
    The difference between the smoothmax of all incorrect logits and the logit
    corresponding to the ground-truth label.
    """

    if calculate_base_loss.base_loss in ['diff', 'sdiff']:  # type: ignore
        correct_logits = logits.gather(1, labels[:, None]).squeeze(1)
        if abstain:
            correct_logits, _ = torch.stack([
                correct_logits, logits[:, -1] + math.log(1 - abstain_cost),
            ]).max(0)

        num_classes = logits.size()[1]
        logits_no_abstain = logits
        if abstain:
            num_classes -= 1
            logits_no_abstain = logits_no_abstain[:, :-1]

        logit_indices = torch.arange(
            num_classes,
            dtype=labels.dtype,
            device=labels.device,
        )[None, :].expand(labels.size()[0], -1)
        incorrect_logits = torch.where(
            logit_indices == labels[:, None],
            torch.full_like(logits_no_abstain, float('-inf')),
            logits_no_abstain,
        )

        if calculate_base_loss.base_loss == 'diff':  # type: ignore
            max_incorrect_logits, _ = torch.max(
                incorrect_logits, 1)
        elif calculate_base_loss.base_loss == 'sdiff':  # type: ignore
            incorrect_probs = incorrect_logits.softmax(1)
            incorrect_logits = torch.where(
                logit_indices == labels[:, None],
                torch.full_like(logits, 0),
                logits,
            )
            max_incorrect_logits = torch.sum(
                incorrect_probs * incorrect_logits, 1)

        return max_incorrect_logits - correct_logits
    elif calculate_base_loss.base_loss == 'ce':  # type: ignore
        return ce_loss_with_abstain(logits, labels, abstain, reduction='none')
    else:
        raise ValueError('Invalid base loss')


calculate_base_loss.base_loss = 'diff'  # type: ignore


def build_gradient_difference(model, normalizer, base_grads, inputs, labels):
    logits = model(normalizer(inputs))
    adv_logit_diff = calculate_base_loss(logits, labels)
    grads, = torch.autograd.grad(adv_logit_diff.sum(), inputs,
                                 create_graph=True)
    return nn.MSELoss()(grads, base_grads)


def calculate_accuracy(logits, labels):
    correct = logits.argmax(1) == labels
    return correct.sum().type(torch.float) / len(correct)


def activation(x, activations=None):
    """
    Returns either the softplus or ReLU of x depending on the value of
    activation.activation. If activations is not None, appends both the
    input and output of the activation to it.
    """

    if activations is not None:
        activations.append(x)

    if activation.activation == 'softplus':  # type: ignore
        x = F.softplus(x)
    elif activation.activation == 'relu':  # type: ignore
        x = F.relu(x)
    else:
        raise NotImplementedError(
            f'activation {activation.activation} '  # type: ignore
            'not supported',
        )

    if activations is not None:
        activations.append(x)

    return x


activation.activation = 'softplus'  # type: ignore


def normalize(x, p=2):
    """
    Normalizes all elements of x (as sliced along the 0th dimension) to have
    lp norm of exactly 1.
    """

    norm = x.norm(
        p=p, dim=list(range(1, len(x.size()))))
    while len(norm.size()) < len(x.size()):
        norm = norm[:, None]
    return x / (norm + 1e-8)


def project(x, y, num_batch_dims=None):
    """
    Project x onto y. x can be of the same shape as y or it can be
    a batch.
    """

    if num_batch_dims == 0:
        num_batch_dims = len(x.size()) - len(y.size())
    non_batch_dims = list(range(num_batch_dims, len(x.size())))
    y = y[(None,) * (len(x.size()) - len(y.size()))]

    factors = ((x * y).sum(dim=non_batch_dims) /
               (1e-8 + (y * y).sum(dim=non_batch_dims)))
    factors = factors[(slice(None),) * num_batch_dims +
                      (None,) * len(non_batch_dims)]
    return factors * y
