import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLRWarmup(_LRScheduler):
    """
    Cosine Annealing with Warm Up learning rate scheduler
    """
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of iterations
            T_warmup: Warmup iterations
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            # Warm up phase
            return [base_lr * (self.last_epoch + 1) / self.T_warmup 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            return [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) /
                                (self.T_max - self.T_warmup))) / 2
                   for base_lr in self.base_lrs]