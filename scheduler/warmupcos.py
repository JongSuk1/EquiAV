import math

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        final_lr,
        T_max,
        last_epoch=-1
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        # print(f'lr: {new_lr} / step: {self._step} / start_lr: {self.start_lr} / ref_lr: {self.ref_lr} / final_lr: {self.final_lr} / progress: {progress} / T_max: {self.T_max} / warmup_steps: {self.warmup_steps}')

        self._step += 1

        return new_lr
    
def Scheduler(optimizer, warmup_steps, start_lr, ref_lr, final_lr, T_max, **kwargs):

	sche_fn = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=warmup_steps, start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr, T_max=T_max)

	return sche_fn