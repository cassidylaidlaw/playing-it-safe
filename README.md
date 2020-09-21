# Playing It Safe: Adversarial Robustness with an Abstain Option

This repository contains code for the paper [Playing it Safe: Adversarial Robustness with an Abstain Option](https://arxiv.org/abs/1911.11253).

## Combined Abstention Robustness Learning (CARL)

This section describes how to train a robust classifier using Combined Abstention Robustness Learning (CARL).

The code requires Python 3.7. First, install the requirements using pip:

    pip install -r requirements.txt

Now, you can train a model with CARL and it will be evaluated at the end of training. To train a model on MNIST, run

    # Change $LAMBDA to the desired value for lambda. To use the ℓ(1) adversarial loss, set
    # $VARIANT to sum; to use the ℓ(2) loss, set $VARIANT to prodlog.
    python3.7 train.py \
      --arch "BigMnistNet" \
      --dataset mnist \
      --loss "l_ce + $LAMBDA*l_adv_abstain(lp=math.inf, pgd_variant=('abstain', 'sum', 'interp'), variant='$VARIANT') + 0.02*l_grad_l1" \
      --abstain \
      --batch_size 100 \
      --num_epochs 40 \
      --lr_drop_epoch 30 \
      --optim adam \
      --val_batches 5 \
      --keep_every 10 \
      --log_dir logs

To train a model on CIFAR10, run

    # Change $LAMBDA to the desired value for lambda. To use the ℓ(1) adversarial loss, set
    # $VARIANT to sum; to use the ℓ(2) loss, set $VARIANT to prodlog.
    python3.7 train.py \
      --arch "WideResNet28x(5)" \
      --dataset cifar10 \
      --loss "l_ce + $LAMBDA*l_adv_abstain(lp=math.inf, pgd_variant=(None, 'interp'), variant='$VARIANT') + 0.02*l_grad_l1" \
      --abstain \
      --batch_size 50 \
      --num_epochs 60 \
      --optim adam \
      --val_batches 5 \
      --keep_every 10 \
      --log_dir logs
