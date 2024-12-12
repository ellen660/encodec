import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing after warmup
            lr_scale = 0.5 * (1 + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))

        # Return adjusted learning rates for all parameter groups
        return [self.min_lr + (base_lr - self.min_lr) * lr_scale for base_lr in self.base_lrs]

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, base_scheduler: _LRScheduler, last_epoch: int = -1):
        """
        WarmupScheduler: Linearly increases the learning rate for the first `warmup_steps` steps,
        then applies the `base_scheduler` afterwards.

        Args:
            optimizer (Optimizer): The optimizer being used (e.g., Adam, SGD).
            warmup_steps (int): The number of warmup steps.
            base_scheduler (_LRScheduler): A learning rate scheduler to apply after warmup (e.g., CosineAnnealingLR).
            last_epoch (int): The index of the last epoch. Used for resuming training.
        """
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # After warmup, defer to the base scheduler
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            super(WarmupScheduler, self).step(epoch)
        else:
            # Step the base scheduler after warmup
            self.base_scheduler.step(epoch)

# # Example usage
# model = torch.nn.Linear(10, 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# # Define the base scheduler (e.g., cosine annealing after warmup)
# base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# # Define the warmup scheduler (e.g., 10 steps of warmup)
# warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=10, base_scheduler=base_scheduler)

# # Training loop
# for epoch in range(100):
#     # Training code here
#     # ...

#     # Step the scheduler at the end of the epoch
#     warmup_scheduler.step()
#     print(f"Epoch {epoch + 1}: Learning Rate = {optimizer.param_groups[0]['lr']}")
