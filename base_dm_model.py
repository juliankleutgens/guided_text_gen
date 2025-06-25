# base_ft_model.py
import torch
import lightning as L
import dataloader
import typing


class BaseDMModel(L.LightningModule):
    """
    Fault-tolerant Lightning mix-in that bundles every utility
    shared by Diffusion, RatioEstimator and the classifier.

    The child class must set _before_ super().__init__() returns:
        self.config            – Hydra cfg node
        self.diffusion         – "absorbing_state" | "uniform"
        self.vocab_size        – int
        self.mask_index        – int
        self.noise             – noise schedule instance
    """

    # ---------- construction ------------------------------------------------
    def __init__(self):
        super().__init__()
        self.fast_forward_epochs: typing.Union[int, None] = None
        self.fast_forward_batches: typing.Union[int, None] = None
        # present in both training and ratio configs
        tr_cfg = getattr(self.config, "training_ratio", getattr(self.config, "training"))
        self.sampling_eps = tr_cfg.sampling_eps

    # ---------- checkpoint fast-forwarding ----------------------------------
    def on_load_checkpoint(self, checkpoint):
        if getattr(self, "ema", None):
            self.ema.load_state_dict(checkpoint['ema'])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if getattr(self, "ema", None):
            checkpoint['ema'] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
                               'epoch_loop.automatic_optimization.optim_progress'][
                               'optimizer']['step']['total'][
                               'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
                               'epoch_loop.automatic_optimization.optim_progress'][
                               'optimizer']['step']['current'][
                               'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        """
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer. \
                train_dataloader.sampler.state_dict()
            checkpoint['sampler'][
                'random_state'] = sampler_state_dict.get(
                'random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None
        """
        # --- save RNG state of one sampler, if available -----------------------
        train_dl = self.trainer.train_dataloader
        random_state = None

        # CombinedLoader → dict of sub-loaders
        if isinstance(train_dl, dict):
            for sub_dl in train_dl.values():
                if hasattr(sub_dl, "sampler") and hasattr(sub_dl.sampler, "state_dict"):
                    random_state = sub_dl.sampler.state_dict().get("random_state", None)
                    break
        else:
            if hasattr(train_dl, "sampler") and hasattr(train_dl.sampler, "state_dict"):
                random_state = train_dl.sampler.state_dict().get("random_state", None)

        checkpoint.setdefault("sampler", {})["random_state"] = random_state

    # ---------- dataloader re-wiring ---------------------------------------
    def on_train_start(self):
        """Hook called at the beginning of training."""
        if getattr(self, "ema", None):
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
                self.trainer._accelerator_connector.use_distributed_sampler
                and self.trainer._accelerator_connector.is_distributed)
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(
                    dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (distributed
                    and self.fast_forward_epochs is not None
                    and self.fast_forward_batches is not None):
                dl_sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': (self.fast_forward_batches
                                * self.config.loader.batch_size)})
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    # ---------- corruption helpers -----------------------------------------
    def _q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape
            (batch_size, 1).
        """
        move_indices = torch.rand(
            *x.shape, device=x.device) < move_chance
        if self.config.diffusion == 'absorbing_state':
            return torch.where(move_indices, self.mask_index, x)
        if self.config.diffusion == 'uniform':
            uniform_tensor = torch.randint(
                0, self.vocab_size, x.shape, device=x.device)
            return torch.where(move_indices, uniform_tensor, x)
        raise NotImplementedError(
            f'Diffusion type {self.config.diffusion} not '
            'implemented.')

    # ---------- time-sampling helper ---------------------------------------
    def _sample_t(self, n):
        """Return n timesteps in (0,1] with the same rules every model uses."""
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _get_time_conditioning_and_move_chance(self, t):
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += (1 / self.T)
        if self.change_of_variables:
            time_conditioning = t[:, None]
            f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
            f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, _ = self.noise(t)
            time_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])
        return time_conditioning, move_chance