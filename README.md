# Playing It Safe: Adversarial Robustness with an Abstain Option

This repository contains code for the paper [Playing it Safe: Adversarial Robustness with an Abstain Option](https://arxiv.org/abs/1911.11253).

The code requires Python 3.7. Install the requirements using pip:

    pip install -r requirements.txt

## Combined Abstention Robustness Learning (CARL)

This section describes how to train a robust classifier using Combined Abstention Robustness Learning (CARL).

To train a model on MNIST, run

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

To train a model on CIFAR-10, run

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

## Natural and adversarial error evaluation 

To evaluate the natural and adversarial error of a trained classifier (like the results in Tables 1 and 4), use the `evaluate.py` script. For instance, the following command will evaluate a trained model on CIFAR-10:

    # Replace /path/to/ckpt.pth with the path to your trained model checkpoint.
    python3.7 evaluate.py \
      --arch "WideResNet28x(5)" \
      --dataset cifar10 \
      --checkpoint /path/to/ckpt.pth \
      --abstain \
      pgd_linf pgd_abstain_sum_linf \
      pgd_abstain_interp_linf pgd_abstain_switch_linf \
      deepfool_linf deepfool_abstain_linf
