from typing import Callable, Optional
from typing_extensions import Protocol
import torch
import math
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

import utils


class Attack(Protocol):
    __name__: str
    def __call__(
        self,
        model: nn.Module,
        normalizer: Callable[[torch.Tensor], torch.Tensor],
        inputs: torch.Tensor,
        labels: torch.Tensor,
        dataset: str,
        abstain: bool=False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        pass


def none(
    model: nn.Module,
    normalizer: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    labels: torch.Tensor,
    dataset: str,
    abstain: bool=False,
) -> torch.Tensor:
    return inputs


PGD_ITERS = 10
# use more iters for validation
PGD_ITERS_VAL = 100
EPS = {
    'mnist2': {
        2: 1,
        math.inf: 0.3,
    },
    'mnist': {
        2: 1,
        math.inf: 0.3,
    },
    'svhn': {
        2: 1,
        math.inf: 12 / 255,
    },
    'cifar10': {
        2: 1,
        math.inf: 8 / 255,
    },
}


def pgd(
    model: nn.Module,
    normalizer: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    labels: torch.Tensor,
    dataset: str,
    abstain: bool=False,
    lp: float=math.nan,
    eps: float=None,
    iters: int=PGD_ITERS,
    initial_perturbations: torch.Tensor=None,
    eps_iter: float=None,
    eps_decay: float=1,
) -> torch.Tensor:
    if eps is None:
        eps = EPS[dataset][lp]

    def loss_fn(adv_inputs):
        logits = model(normalizer(adv_inputs))
        return utils.calculate_base_loss(logits, labels, abstain)

    if eps_iter is None:
        eps_iter = eps * 2 / iters

    return iterated_pgd(inputs, loss_fn, dataset, lp, eps,
                        iters=iters, eps_iter=eps_iter, eps_decay=eps_decay,
                        initial_perturbations=initial_perturbations,
                        keep_best=True)


def pgd_l2(*args, **kwargs):
    kwargs['lp'] = 2
    kwargs.setdefault('iters', PGD_ITERS_VAL)
    return pgd(*args, **kwargs)


def pgd_linf(*args, **kwargs):
    kwargs['lp'] = math.inf
    kwargs.setdefault('iters', PGD_ITERS_VAL)
    return pgd(*args, **kwargs)


def _pgd_abstain_loss_fn(variant, iters):
    def sum_loss_fn(logits, labels):
        abstain_logits = logits[:, -1]
        non_abstain_logits = logits[:, :-1]
        return (
            utils.calculate_base_loss(non_abstain_logits, labels) -
            abstain_logits
        )

    def cw_loss_fn(logits, labels):
        return utils.calculate_base_loss(logits, labels, abstain=False)

    def abstain_loss_fn(logits, labels):
        return utils.calculate_base_loss(logits, labels, abstain=True)

    current_iter = 0

    def switch_loss_fn(logits, labels):
        nonlocal current_iter
        if current_iter < iters // 2:
            loss = cw_loss_fn(logits, labels)
        else:
            loss = abstain_loss_fn(logits, labels)
        current_iter += 1
        return loss

    def interp_loss_fn(logits, labels):
        nonlocal current_iter
        weight = current_iter / (iters - 1)
        loss = (
            (1 - weight) * cw_loss_fn(logits, labels) +
            weight * abstain_loss_fn(logits, labels)
        )
        current_iter += 1
        return loss

    return {
        'cw': cw_loss_fn,
        'abstain': abstain_loss_fn,
        'sum': sum_loss_fn,
        'switch': switch_loss_fn,
        'interp': interp_loss_fn,
    }[variant]


def pgd_abstain(model, normalizer, inputs, labels, dataset, lp, eps=None,
                iters=PGD_ITERS, abstain=True, variant='sum',
                stop=False, **kwargs):
    """
    Special variants of PGD for abstain setting. They balance between avoiding
    the correct and abstain classes differently.
     - variant='sum': loss is (max f_i(x), i != y, A) - (f_y(x) + f_A(x))
     - variant='switch': loss is (max f_i(x), i != y) - f_y(x) for half the
       iterations, then  (max f_i(x), i != y, A) - max(f_y(x), f_A(x))
     - variant='interp': like 'switch' but interpolates linearly from the first
       loss to the second
    """

    if not abstain:
        return inputs

    if eps is None:
        eps = EPS[dataset][lp]

    eps_iter = eps * 2 / iters
    logits_loss_fn = _pgd_abstain_loss_fn(variant, iters)

    def loss_fn(adv_inputs):
        logits = model(normalizer(adv_inputs))
        pred_labels = logits.argmax(1)
        abstain_class = logits.size()[1] - 1
        success = (pred_labels != labels) & (pred_labels != abstain_class)

        loss = logits_loss_fn(logits, labels)
        if stop:
            loss[success] = 1e8
        return loss

    if variant in ['sum', 'abstain']:
        kwargs.setdefault('initial_perturbations',
                          torch.zeros_like(inputs))
        kwargs.setdefault('keep_best', True)

    return iterated_pgd(
        inputs, loss_fn, dataset, lp, eps,
        iters=iters, eps_iter=eps_iter, **kwargs,
    )


def pgd_abstain_sum_linf(*args, **kwargs):
    kwargs['lp'] = math.inf
    kwargs.setdefault('iters', PGD_ITERS_VAL)
    kwargs['variant'] = 'sum'
    kwargs.setdefault('stop', True)
    return pgd_abstain(*args, **kwargs)


def pgd_abstain_switch_linf(*args, **kwargs):
    kwargs['lp'] = math.inf
    kwargs.setdefault('iters', PGD_ITERS_VAL)
    kwargs['variant'] = 'switch'
    kwargs.setdefault('stop', True)
    return pgd_abstain(*args, **kwargs)


def pgd_abstain_interp_linf(*args, **kwargs):
    kwargs['lp'] = math.inf
    kwargs.setdefault('iters', PGD_ITERS_VAL)
    kwargs['variant'] = 'interp'
    kwargs.setdefault('stop', True)
    return pgd_abstain(*args, **kwargs)

def pgd_abstain_interp_l2(*args, **kwargs):
    kwargs['lp'] = 2
    kwargs.setdefault('iters', PGD_ITERS_VAL)
    kwargs['variant'] = 'interp'
    kwargs.setdefault('stop', True)
    return pgd_abstain(*args, **kwargs)


def noise(model, normalizer, inputs, labels, dataset,
          abstain=False):
    """Returns random noise."""
    return torch.rand_like(inputs)


def noise_pgd(model, normalizer, inputs, labels, dataset, *args, **kwargs):
    noise_inputs = noise(model, normalizer, inputs, labels, dataset)
    return pgd(model, normalizer, noise_inputs, labels, dataset, *args,
               **kwargs)


def noise_pgd_linf(*args, **kwargs):
    return noise_pgd(*args, lp=math.inf, iters=PGD_ITERS_VAL, **kwargs)


def deepfool(model, normalizer, inputs, labels, dataset, lp, eps=None,
             iters=50, constrain_eps=True, abstain=False, overshoot=0.02,
             num_targets=math.inf):
    if constrain_eps and eps is None:
        eps = EPS[dataset][lp]

    logits = model(normalizer(inputs))
    logits_no_abstain = logits

    num_classes = logits.size()[1]
    if abstain:
        logits_no_abstain = logits_no_abstain[:, :-1]
        num_classes -= 1

    inputs = inputs.clone().detach().to(torch.float)
    perturbations: torch.Tensor = torch.zeros_like(inputs)

    for _ in range(iters):
        adv_inputs = inputs + (1 + overshoot) * perturbations
        adv_inputs.requires_grad_(True)
        zero_gradients(adv_inputs)
        logits = model(normalizer(adv_inputs))

        logit_indices = torch.arange(
            num_classes,
            dtype=labels.dtype,
            device=labels.device,
        )[None, :].expand(labels.size()[0], -1)
        incorrect_logits = torch.where(
            logit_indices == labels[:, None],
            torch.full_like(logits_no_abstain, -math.inf),
            logits_no_abstain,
        )
        target_classes = incorrect_logits.argsort(
            dim=1, descending=True)[:, :min(num_classes - 1, num_targets)]

        classified_labels = logits.argmax(1)
        live = classified_labels == labels
        if abstain:
            live = live | (classified_labels == num_classes)

        if torch.all(~live):
            # Stop early if all inputs are already misclassified
            break

        smallest_magnitudes = torch.full(
            (int(live.sum()),), math.inf,
            dtype=torch.float, device=perturbations.device)
        smallest_perturbation_updates = torch.zeros_like(perturbations[live])

        if target_classes.size()[1] > 1:
            logits[live, classified_labels[live]].sum().backward(retain_graph=True)
            grads_correct = adv_inputs.grad.data[live].clone().detach()

        for k in range(target_classes.size()[1]):
            zero_gradients(adv_inputs)

            if target_classes.size()[1] > 1:
                logits_target = logits[live, target_classes[live, k]]
                logits_target.sum().backward(retain_graph=True)
                grads_target = adv_inputs.grad.data[live].clone().detach()

                grads_diff = (grads_target - grads_correct).detach()
                logits_margin = (logits_target -
                                 logits[live,
                                        classified_labels[live]]).detach()
            else:
                logits_margin = (logits[live, target_classes[live, k]] -
                                 logits[live, classified_labels[live]])
                logits_margin.sum().backward()
                grads_diff = adv_inputs.grad.data[live].detach()
                logits_margin = logits_margin.detach()

            grads_norm = grads_diff.norm(
                p=1 if lp == math.inf else 2,
                dim=list(range(1, len(grads_diff.size()))))
            magnitudes = logits_margin.abs() / grads_norm

            magnitudes_expanded = magnitudes
            for _ in range(len(grads_diff.size()) - 1):
                grads_norm = grads_norm.unsqueeze(-1)
                magnitudes_expanded = magnitudes_expanded.unsqueeze(-1)

            if lp == math.inf:
                perturbation_updates = ((magnitudes_expanded + 1e-4) *
                                        torch.sign(grads_diff))
            else:
                perturbation_updates = ((magnitudes_expanded + 1e-4) *
                                        grads_diff / grads_norm)

            smaller = magnitudes < smallest_magnitudes
            smallest_perturbation_updates[smaller] = \
                perturbation_updates[smaller]
            smallest_magnitudes[smaller] = magnitudes[smaller]

        all_perturbation_updates = torch.zeros_like(perturbations)
        all_perturbation_updates[live] = smallest_perturbation_updates
        perturbations.add_(all_perturbation_updates)

        perturbations = perturbations * (1 + overshoot)
        _project_perturbations(perturbations, inputs, eps, lp)
        perturbations = perturbations / (1 + overshoot)

    adv_inputs = inputs + perturbations * (1 + overshoot)

    return adv_inputs


def deepfool_linf(*args, **kwargs):
    kwargs['lp'] = math.inf
    return deepfool(*args, **kwargs)


def deepfool_l2(*args, **kwargs):
    kwargs['lp'] = 2
    return deepfool(*args, **kwargs)


def deepfool_abstain_linf(model, normalizer, inputs, labels, dataset,
                          eps=None, iters=50, constrain_eps=True,
                          abstain=True, overshoot=0.02, num_targets=math.inf):
    """
    Variant of DeepFool that finds closest point in Linf distance that crosses
    both correct and abstain decision boundaries at each step assuming
    linearity.
    """

    assert abstain is True

    if constrain_eps and eps is None:
        eps = EPS[dataset][math.inf]

    inputs = inputs.clone().detach().to(torch.float)
    perturbations = torch.zeros_like(inputs)

    for _ in range(iters):
        adv_inputs = inputs + (1 + overshoot) * perturbations
        adv_inputs.requires_grad_(True)
        zero_gradients(adv_inputs)
        logits = model(normalizer(adv_inputs))
        logits_no_abstain = logits

        num_classes = logits.size()[1]
        if abstain:
            logits_no_abstain = logits_no_abstain[:, :-1]
            num_classes -= 1

        logit_indices = torch.arange(
            num_classes,
            dtype=labels.dtype,
            device=labels.device,
        )[None, :].expand(labels.size()[0], -1)
        incorrect_logits = torch.where(
            logit_indices == labels[:, None],
            torch.full_like(logits_no_abstain, -math.inf),
            logits_no_abstain,
        )
        target_classes = incorrect_logits.argsort(
            dim=1, descending=True)[:, :min(num_classes - 1, num_targets)]

        classified_labels = logits.argmax(1)
        live = classified_labels == labels
        if abstain:
            live = live | (classified_labels == num_classes)

        if torch.all(~live):
            # Stop early if all inputs are already misclassified
            break

        smallest_magnitudes = torch.full(
            (int(live.sum()),), math.inf,
            dtype=torch.float, device=perturbations.device)
        smallest_perturbation_updates = torch.zeros_like(perturbations[live])

        if target_classes.size()[1] > 1:
            zero_gradients(adv_inputs)
            logits[live, classified_labels[live]].sum().backward(
                retain_graph=True)
            grads_correct = adv_inputs.grad.data[live].clone().detach()

            zero_gradients(adv_inputs)
            logits[live, -1].sum().backward(retain_graph=True)
            grads_abstain = adv_inputs.grad.data[live].clone().detach()

        for k in range(target_classes.size()[1]):
            if target_classes.size()[1] > 1:
                zero_gradients(adv_inputs)
                logits_target = logits[live, target_classes[live, k]]
                logits_target.sum().backward(retain_graph=True)
                grads_target = adv_inputs.grad.data[live].clone().detach()

                grads_diff_correct = (grads_target - grads_correct).detach()
                logits_margin_correct = (
                    logits_target -
                    logits[live, labels[live]]
                ).detach()

                grads_diff_abstain = (grads_target - grads_abstain).detach()
                logits_margin_abstain = (logits_target -
                                         logits[live, -1]).detach()
            else:
                zero_gradients(adv_inputs)
                logits_margin_correct = (
                    logits[live, target_classes[live, k]] -
                    logits[live, labels[live]]
                )
                logits_margin_correct.sum().backward(retain_graph=True)
                grads_diff_correct = adv_inputs.grad.data[live].detach()
                logits_margin_correct = logits_margin_correct.detach()

                zero_gradients(adv_inputs)
                logits_margin_abstain = (logits[live, target_classes[live, k]]
                                         - logits[live, -1])
                logits_margin_abstain.sum().backward(retain_graph=True)
                grads_diff_abstain = adv_inputs.grad.data[live].detach()
                logits_margin_abstain = logits_margin_abstain.detach()

            assert torch.all((logits_margin_correct < 0) |
                             (logits_margin_abstain < 0))

            grads_diff = torch.stack([
                grads_diff_correct.view(live.sum(), -1),
                grads_diff_abstain.view(live.sum(), -1),
            ], dim=1)
            logits_margin = torch.stack([
                logits_margin_correct,
                logits_margin_abstain,
            ], dim=1)

            # First, see if we can just cross both decision boundaries in one
            # step
            for logits_margin_boundary, grads_diff_boundary in [
                (logits_margin_correct, grads_diff_correct),
                (logits_margin_abstain, grads_diff_abstain),
            ]:
                grads_diff_boundary = grads_diff_boundary.view(live.sum(), -1)
                grads_norm = grads_diff_boundary.norm(p=1, dim=1)
                magnitudes = logits_margin_boundary.abs() / grads_norm
                perturbation_updates = (
                    (magnitudes + 1e-4)[:, None] *
                    grads_diff_boundary.sign()
                )
                resulting_margins = (
                    grads_diff.matmul(perturbation_updates[:, :, None])
                    [:, :, 0] + logits_margin
                )
                single_boundary_success = resulting_margins.min(1).values >= 0
                smaller = (
                    (magnitudes < smallest_magnitudes) &
                    single_boundary_success
                )
                perturbation_updates = perturbation_updates \
                    .reshape(smallest_perturbation_updates.size())
                smallest_perturbation_updates[smaller] = \
                    perturbation_updates[smaller]
                smallest_magnitudes[smaller] = magnitudes[smaller]

            # Algorithm taken from "A Finite Algorithm for the Minimum l∞
            # Solution to a System of Consistent Linear Equations" by Cadzow
            A = grads_diff
            y = -logits_margin

            # step 2
            v = torch.stack([-A[:, 1], A[:, 0]], dim=1)

            # step 3
            sgn_yv = (y[:, :, None] * v).sum(1).sign()
            A2v1 = (A.transpose(1, 2).matmul(v) *
                    (1 - torch.eye(A.size()[2], device=A.device))).abs().sum(2)
            u = (sgn_yv / A2v1)[:, None] * v

            # step 4
            yu = (y[:, :, None] * u).sum(1)

            # step 5
            A2u = A.transpose(1, 2).matmul(u) * (1 - torch.eye(A.size()[2], device=A.device))
            x1 = (y[:, 0, None] -
                  yu * A[:, 0:1].matmul(A2u.sign())[:, 0]) / A[:, 0]

            # step 6
            i = (x1.abs() < yu).long().argmax(1)
            x = A2u.gather(2, i[:, None, None].expand(A2u.size())) \
                .sign()[:, :, 0] * yu.gather(1, i[:, None].expand(yu.size()))
            x, x1
            x[range(x.size()[0]), i] = x1[range(x1.size()[0]), i]

            magnitudes = x.abs().max(1).values
            perturbation_updates = x \
                .reshape(smallest_perturbation_updates.size())

            smaller = magnitudes < smallest_magnitudes
            smallest_perturbation_updates[smaller] = \
                perturbation_updates[smaller]
            smallest_magnitudes[smaller] = magnitudes[smaller]

        all_perturbation_updates = torch.zeros_like(perturbations)
        all_perturbation_updates[live] = smallest_perturbation_updates
        perturbations.add_(all_perturbation_updates)

        perturbations = perturbations * (1 + overshoot)
        _project_perturbations(perturbations, inputs, eps, lp=math.inf)
        perturbations = perturbations / (1 + overshoot)

    adv_inputs = inputs + perturbations * (1 + overshoot)

    return adv_inputs


def deepfool_unconstrained(model, normalizer, inputs, labels, dataset,
                           lp, iters=10):
    perturbations = torch.zeros_like(inputs)
    adv_examples = inputs + perturbations

    for _ in range(iters):
        adv_examples.requires_grad = True
        logits = model(normalizer(adv_examples))
        base_loss = utils.calculate_base_loss(logits, labels)
        base_loss.sum().backward()
        grads = adv_examples.grad
        if lp == math.inf:
            grad_norms = grads.reshape(grads.size()[0], -1).norm(
                dim=1, p=1)
            grads = grads.sign()
        else:
            grad_norms = grads.reshape(grads.size()[0], -1).norm(
                dim=1, p=2)
            grads = utils.normalize(grads, 2)
        step_factor = (
            (base_loss / (grad_norms + 1e-8))
            [(slice(None),) + (None,) * len(adv_examples.size()[:-1])]
        )
        adv_examples = (adv_examples - step_factor * grads).detach()

    return adv_examples


def _project_perturbations(perturbations, inputs, eps, lp):
    if lp == math.inf:
        clipped_perturbations = \
            torch.clamp(perturbations / eps, -1, 1) * eps
    else:
        clipped_perturbations = (perturbations / eps).renorm(
            p=lp, dim=0, maxnorm=1) * eps

    new_image = torch.clamp(inputs + clipped_perturbations,
                            0, 1)
    perturbations.data.add_((new_image - inputs) - perturbations)


def iterated_pgd(
    inputs: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    dataset: str,
    lp: float,
    eps: Optional[float]=None,
    iters: int=PGD_ITERS,
    eps_iter: float=None,
    initial_perturbations=None,
    eps_decay=1,
    inverse_eps_decay=False,
    early_stop_thresh=0,
    keep_best=False,
) -> torch.Tensor:
    if eps is None:
        eps = EPS[dataset][lp]

    perturbations: torch.Tensor
    if initial_perturbations is None:
        perturbations = torch.zeros_like(inputs)
        perturbations.uniform_(-1, 1)
        perturbations.mul_(eps)
    else:
        perturbations = initial_perturbations.detach().clone()

    _project_perturbations(perturbations, inputs, eps, lp)
    perturbations.requires_grad = True
    optimizer = optim.Adam([perturbations], lr=10)

    last_loss = None

    best_loss = None
    best_adv_inputs = inputs.detach().clone()

    for attack_iter in range(iters):
        optimizer.zero_grad()

        # calculate loss
        adv_inputs = inputs + perturbations
        adv_loss = -loss_fn(adv_inputs)  # we want to increase the loss

        # update best adversarial inputs so far
        if adv_loss.size() == inputs.size()[0:1]:
            if best_loss is None:
                best_loss = adv_loss
                best_adv_inputs = adv_inputs.detach().clone()
            else:
                best_so_far = adv_loss < best_loss
                best_loss[best_so_far] = adv_loss.detach()[best_so_far]
                best_adv_inputs[best_so_far] = adv_inputs.detach()[best_so_far]
        adv_loss = adv_loss.mean()

        adv_loss.backward(retain_graph=True)

        if inverse_eps_decay:
            eps_iter = eps / (attack_iter + 5)

        # optimize
        if eps_iter is None:
            optimizer.step()
        else:
            grads = perturbations.grad.data
            if lp == math.inf:
                grads = grads.sign()
            else:
                grads = utils.normalize(grads, lp)
            perturbations.data.add_(-eps_iter * grads)

        eps_iter = eps_iter * eps_decay

        _project_perturbations(perturbations, inputs, eps, lp)

        if last_loss is not None:
            if torch.abs(adv_loss - last_loss) < early_stop_thresh:
                break
        last_loss = adv_loss.detach()

    if keep_best:
        return best_adv_inputs
    else:
        return inputs + perturbations.detach()
