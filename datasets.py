
import torch
from torchvision.datasets import MNIST


class TwoClassMNIST(MNIST):
    """
    MNIST restricted to just two classes (by default 6s and 7s as in the
    unrestricted adversarial examples challenge).
    """

    data: torch.Tensor
    targets: torch.Tensor

    def __init__(self, *args, class0=6, class1=7, **kwargs):
        super().__init__(*args, **kwargs)

        indices = ((self.targets == class0) |
                   (self.targets == class1))
        self.data = self.data[indices]
        self.targets = torch.where(
            self.targets[indices] == class0,
            torch.zeros(indices.sum()),
            torch.ones(indices.sum()),
        ).type(torch.long)
