
import math
from typing import Callable
import torch
from torch import nn

import attacks
import random


class Loss(object):
    """
    A particular loss term that can be constructed and differentiated at each
    iteration for the current model and batch.
    """

    def __init__(self, model, normalizer, inputs, labels, dataset,
                 log_fn, train,
                 inputs_to_compute, computed_logits):
        self.model = model
        self.normalizer = normalizer
        self.inputs = inputs
        self.labels = labels
        self.dataset = dataset
        self.log_fn = log_fn
        self.train = train

        self.inputs_to_compute = inputs_to_compute
        self.computed_logits = computed_logits

        self.in_setup = True

    def add_input_to_compute(self, inputs):
        self.inputs_to_compute.append(inputs)

    def get_orig_logits(self):
        return self.computed_logits[0]

    def get_computed_logits(self):
        return self.computed_logits.pop(1)

    def setup(self, *args, **kwargs) -> None:
        pass

    def calculate(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def switch_to_eval(self):
        self.in_setup = False

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if self.in_setup:
            self.setup(*args, **kwargs)
            return torch.ones(1) * math.nan
        else:
            return self.calculate(*args, **kwargs)

    @staticmethod
    def reduce(x: torch.Tensor, reduction='mean') -> torch.Tensor:
        """
        Given a tensor x, either returns:
         * the batchwise mean of x (if reduction is 'none')
         * the mean of x (if reduction is 'mean')
         * the sum of x (if reduction is 'sum')
        """

        if reduction == 'none':
            if len(x.size()) <= 1:
                return x
            else:
                return torch.mean(x, dim=list(range(1, len(x.size()))))
        elif reduction == 'mean':
            return torch.mean(x)
        elif reduction == 'sum':
            return torch.sum(x)
        else:
            raise ValueError(f'invalid reduction {repr(reduction)}')


class AdversarialLoss(Loss):
    """
    Loss term that is the cross-entropy between the correct labels and the
    model output for adversarial examples for each input.
    """

    def setup(self, lp, eps=None):
        adv_inputs = attacks.pgd(
            self.model, self.normalizer, self.inputs, self.labels,
            self.dataset, lp, eps,
        )
        self.add_input_to_compute(adv_inputs)

    def calculate(self, lp, eps=None, reduction='mean'):
        adv_logits = self.get_computed_logits()

        ce = nn.CrossEntropyLoss(reduction=reduction)(adv_logits, self.labels)
        if ce.numel() == 1:
            self.log_fn('loss/adv', ce.item())
        return ce


class AdversarialAbstainLoss(Loss):
    """
    Like AdversarialLoss, but attempts to make the classifier always either
    abstain or correct within an Lp ball.

    If exclusive is True, then the loss is calculated as
        -log((abstain prob) + (correct prob))
    If exclusive is False, then the loss is calculated as
        -log((abstain prob) + (correct prob) - (abstain prob)(correct prob))
    """

    def setup(self, lp, eps=None, abstain_cost=0, deepfool=False,
              pgd_variant=None, decay=False, variant='prodlog',
              adaptive=False, gamma=None, only_correct=False,
              eip=None, stop=False, randomize=True,
              iters=attacks.PGD_ITERS):
        if eps is None:
            eps = attacks.EPS[self.dataset][lp]

        if deepfool:
            adv_inputs = attacks.deepfool(
                self.model, self.normalizer, self.inputs, self.labels,
                self.dataset, lp, eps, abstain=True, num_targets=1,
                iters=iters,
            )
        else:
            if pgd_variant is None or isinstance(pgd_variant, str):
                pgd_variants = [pgd_variant]
            elif randomize:
                pgd_variants = [random.choice(pgd_variant)]
            else:
                pgd_variants = pgd_variant

            adv_inputs = []
            batch_size = len(self.labels)
            for variant_index, pgd_variant in enumerate(pgd_variants):
                attack_kwargs = {'iters': iters}

                if eip is not None:
                    eip_variant, prop_iters, eip_prop = eip
                    if (
                        pgd_variant == eip_variant and
                        random.random() < eip_prop
                    ):
                        attack_kwargs['iters'] = prop_iters

                attack: Callable[..., torch.Tensor]
                if pgd_variant is None:
                    attack = attacks.pgd
                    if 'iters' not in attack_kwargs:
                        attack_kwargs.update({
                            'eps_iter': eps / 2,
                            'eps_decay': (0.8 if attack_kwargs['iters'] >= 10
                                          else 0.6),
                        })
                else:
                    attack = attacks.pgd_abstain
                    attack_kwargs.update({
                        'variant': pgd_variant,
                        'stop': stop,
                    })

                if pgd_variant in ['sum', 'abstain']:
                    attack_kwargs.update({
                        'inverse_eps_decay':
                        attack_kwargs['iters'] == iters,
                        'early_stop_thresh': 0.1,
                        'iters': 100,
                    })

                variant_slice = slice(
                    (variant_index * batch_size) // len(pgd_variants),
                    ((variant_index + 1) * batch_size) // len(pgd_variants),
                )

                adv_inputs.append(attack(
                    self.model, self.normalizer,
                    self.inputs[variant_slice], self.labels[variant_slice],
                    self.dataset, lp=lp, eps=eps, abstain=True,
                    **attack_kwargs,
                ))
            adv_inputs = torch.cat(adv_inputs)

        self.add_input_to_compute(adv_inputs)

    def calculate(self, lp, eps=None, abstain_cost=0, deepfool=False,
                  pgd_variant=None, decay=False, variant='prodlog',
                  adaptive=False, gamma=None, only_correct=False,
                  eip=None, stop=False, iters=attacks.PGD_ITERS,
                  randomize=True, reduction='mean'):
        if eps is None:
            eps = attacks.EPS[self.dataset][lp]

        orig_logits = self.get_orig_logits()
        adv_logits = self.get_computed_logits()

        ce_loss = nn.CrossEntropyLoss(reduction='none')

        logit_indices = torch.arange(
            adv_logits.size()[1],
            dtype=self.labels.dtype,
            device=self.labels.device,
        )[None, :].expand(self.labels.size()[0], -1)
        incorrect_adv_logits = torch.where(
            logit_indices == self.labels[:, None],
            torch.full_like(adv_logits, float('-inf')),
            adv_logits,
        )
        correct_logits = adv_logits.gather(1, self.labels[:, None])[:, 0]
        abstain_logits = adv_logits[:, -1]

        # using LogSumExp is important to ensure numerical stability
        if variant == 'sum':
            loss = (
                adv_logits.logsumexp(1) -
                torch.stack([
                    correct_logits,
                    abstain_logits + math.log(1 - abstain_cost),
                ], dim=1).logsumexp(1)
            )
        elif variant == 'or':
            loss = (
                2 * adv_logits.logsumexp(1) -
                torch.cat([
                    abstain_logits[:, None] + incorrect_adv_logits +
                    math.log(1 - abstain_cost),
                    correct_logits[:, None] + adv_logits,
                ], dim=1).logsumexp(1)
            )
        elif variant in ['prodlog', 'harmlog', 'prod']:
            abstain_class = adv_logits.size()[1] - 1
            correct_loss = ce_loss(adv_logits, self.labels)
            if abstain_cost > 0:
                abstain_loss = (
                    adv_logits.logsumexp(1) -
                    torch.stack([
                        abstain_logits,
                        correct_logits + math.log(abstain_cost),
                    ], dim=1).logsumexp(1)
                )
            else:
                abstain_loss = ce_loss(
                    adv_logits,
                    torch.ones_like(self.labels) * abstain_class,
                )
            if variant == 'prodlog':
                loss = correct_loss * abstain_loss
            elif variant == 'harmlog':
                loss = (correct_loss * abstain_loss + 1e-4).sqrt()
            elif variant == 'prod':
                loss = correct_loss + abstain_loss

        if only_correct:
            pred_labels = orig_logits.argmax(1)
            adv_labels = adv_logits.argmax(1)
            abstain_class = orig_logits.size()[1] - 1
            loss = loss * (
                (pred_labels == self.labels) |
                (adv_labels == self.labels) |
                (adv_labels == abstain_class)
            ).float()

        loss = Loss.reduce(loss, reduction)

        if loss.numel() == 1:
            self.log_fn('loss/adv_abstain', loss.item())
        return loss
