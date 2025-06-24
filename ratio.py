import itertools, typing, torch, torch.nn as nn, torchmetrics
import lightning as L
import hydra.utils
import dataloader, noise_schedule, utils        # unchanged imports
import models
# -------------------------------------------------------------------

class RatioEstimator(L.LightningModule):
    """
    Time-conditioned ratio network r_ψ(x_t, t) with cycle- and
    consistency-regularisation (see TLDM Appendix, pseudo-code 4).
    """
    def __init__(
        self,
        config,
        tokenizer,                       # transformers.PreTrainedTokenizer
        domain_classifier: nn.Module,
        domain_classifier_time_dependent: nn.Module,
        pretrained_backbone: typing.Optional[nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "domain_classifier", "domain_classifier_t",
             "pretrained_backbone",
        ])
        self.config = config            # full Hydra cfg node
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.mask_index = (
            tokenizer.mask_token_id
            if getattr(tokenizer, "mask_token", None) is not None
            else self.vocab_size
        )
        if self.mask_index == self.vocab_size:          # add [MASK]
            self.vocab_size += 1

        # ------------- frozen auxiliary networks --------------------
        self.domain_classifier = domain_classifier.eval().requires_grad_(False)
        self.domain_classifier_t = domain_classifier_time_dependent.eval().requires_grad_(False)
        #self.denoiser_model = denoiser_model.eval().requires_grad_(False)

        # ------------- ratio network backbone -----------------------
        if config.ratio_backbone == "dit":
            self.ratio_model = models.dit.DITRatio(   # ← same arch family
                config, vocab_size=self.vocab_size, time_conditioning=True
            )
        if pretrained_backbone is not None:
            self.ratio_model.load_pretrained_encoder(pretrained_backbone)

        utils.print_num_parameters(self.ratio_model, print_prefix="Ratio model ")

        # ------------- noise schedule & hyper-parameters ------------
        self.noise = noise_schedule.get_noise(config, dtype=self.dtype)
        self.eta1, self.eta2 = config.training_ratio.eta1, config.training_ratio.eta2
        self.diffusion = config.diffusion
        self.T = config.T
        self.sampling_eps = config.training_ratio.sampling_eps
        self.consistency = bool(self.eta2 and self.eta2 > 0)
        self.cycle = bool(self.eta1 and self.eta1 > 0)

        # ------------- logging helpers ------------------------------
        self.mse = nn.MSELoss()
        self.train_metrics = torchmetrics.MetricCollection({"loss": torchmetrics.MeanMetric()}).clone(prefix="train/")
        self.valid_metrics = self.train_metrics.clone(prefix="val/")
    # ================================================================
    # ----------------- checkpointing & fast-forward ----------------
    # ================================================================
    def on_load_checkpoint(self, checkpoint):
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is
        #  1 iteration behind, so we're using the optimizer's
        #  progress.
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
        # _batches_that_stepped tracks the number of global
        # steps, not the number of local steps, so we don't
        # multiply with self.trainer.accumulate_grad_batches
        # here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
            '_batches_that_stepped'] = \
            checkpoint['loops']['fit_loop'][
                'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler,
                   'state_dict'):
            sampler_state_dict = self.trainer. \
                train_dataloader.sampler.state_dict()
            checkpoint['sampler'][
                'random_state'] = sampler_state_dict.get(
                'random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
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
                    persistent_workers=self.config.loader.persistent_workers
                ))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls


    # ================================================================
    #                           helpers
    # ================================================================
    def _q_xt(self, x, move_chance):
        """Apply the absorbing / uniform corruption used in TLDM."""
        move = torch.rand_like(x, dtype=torch.float) < move_chance
        if self.diffusion == "absorbing_state":
            return torch.where(move, self.mask_index, x)
        if self.diffusion == "uniform":
            return torch.where(move, torch.randint(0, self.vocab_size, x.shape, device=x.device), x)
        raise NotImplementedError

    def _sample_t(self, n: int):
        eps = torch.rand(n, device=self.device)
        if self.config.training_ratio.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            eps = (eps / n + offset) % 1
        t = (1 - self.sampling_eps) * eps + self.sampling_eps
        if self.config.training_ratio.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    # ================================================================
    #                       forward interface
    # ================================================================
    def forward(self, x_t, t):
        # x_t: (B, L) int tokens     t: (B,) float in (0,1]
        with torch.cuda.amp.autocast(dtype=torch.float32):
            return self.ratio_model(x_t, t[:, None])  # (B,) logits

    # ================================================================
    #                    core loss computation
    # ================================================================
    def _compute_losses(self, batch):
        x0_src, x0_tgt = batch["input_ids_src"], batch["input_ids_tgt"]
        attention_mask_src, attention_mask_tgt = batch["attention_mask_src"], batch["attention_mask_tgt"]
        B_src = x0_src.size(0)

        # ---------- source branch ----------
        t_src = self._sample_t(B_src)
        sigma_src, _ = self.noise(t_src)
        move_chance_src = 1 - torch.exp(-sigma_src)
        x_t_src = self._q_xt(x0_src, move_chance_src[:, None])
        with torch.no_grad():
            c_src = self.domain_classifier(x0_src, attention_mask=attention_mask_src).squeeze(-1)
            r_src = (-c_src if not self.config.training_ratio.classifier_output_with_sigmoid
                     else torch.log((1 - c_src) / (c_src + 1e-8) + 1e-8))
        r_pred_src = self(x_t_src, t_src)
        loss_ratio = self.mse(r_pred_src, r_src)

        # ---------- cycle loss on target branch ----------
        if self.cycle:
            B_tgt = x0_tgt.size(0)
            t_tgt = self._sample_t(B_tgt)
            sigma_tgt, _ = self.noise(t_tgt)
            move_chance_tgt = 1 - torch.exp(-sigma_tgt)
            x_t_tgt = self._q_xt(x0_tgt, move_chance_tgt[:, None])
            with torch.no_grad():
                c_tdep = self.domain_classifier_t(x_t_tgt, t_tgt).squeeze(-1)
                r_tdep = (-c_tdep if not self.config.training_ratio.classifier_output_with_sigmoid
                          else torch.log((1 - c_tdep) / (c_tdep + 1e-8) + 1e-8))
            r_pred_tgt = self(x_t_tgt, t_tgt)
            loss_cycle = self.mse(r_pred_tgt, r_tdep)
        else:
            loss_cycle = torch.zeros((), device=self.device)

        # ---------- score-consistency loss ----------
        if False:
            x_t_tgt = x_t_tgt.detach().requires_grad_(True)
            log_r = torch.log(self(x_t_tgt, t_tgt) + 1e-20).sum()
            grad_log_r = torch.autograd.grad(log_r, x_t_tgt)[0].detach()
            with torch.no_grad():
                s_src = self.denoiser_model(x_t_src, t_src)
                score_src = -s_src / sigma_src[:, None]
                s_tgt = self.denoiser_model(x_t_tgt, t_tgt)
                score_tgt = -s_tgt / sigma_tgt[:, None]
            grad_target = score_tgt - score_src[:score_tgt.size(0)]
            loss_consistency = self.mse(grad_log_r, grad_target)
        else:
            loss_consistency = torch.zeros((), device=self.device)

        total = loss_ratio + self.eta1 * loss_cycle + self.eta2 * loss_consistency
        return total, loss_ratio, loss_cycle, loss_consistency

    # ================================================================
    #                    Lightning overrides
    # ================================================================
    def training_step(self, batch, _):
        total, l_r, l_cy, l_co = self._compute_losses(batch)
        self.train_metrics.update(torch.tensor([total.detach()]))

        self.log_dict({
            "train/total": total,
            "train/L_ratio": l_r,
            "train/L_cycle": l_cy,
            "train/L_consistency": l_co,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }, prog_bar=True, sync_dist=True)
        return total

    def validation_step(self, batch, _):
        total, l_r, l_cy, l_co = self._compute_losses(batch)
        self.valid_metrics.update(torch.tensor([total.detach()]))
        self.log_dict({
            "val/total": total,
            "val/L_ratio": l_r,
            "val/L_cycle": l_cy,
            "val/L_consistency": l_co,
        }, prog_bar=True, sync_dist=True)

    # ----------------- loader patch identical to Classifier ----------
    def on_train_start(self):
        distributed = (self.trainer._accelerator_connector.use_distributed_sampler
                       and self.trainer._accelerator_connector.is_distributed)
        sampler_cls = (dataloader.FaultTolerantDistributedSampler
                       if distributed else dataloader.RandomFaultTolerantSampler)
        patched = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            sampler = sampler_cls(dl.dataset, shuffle=getattr(dl.sampler, "shuffle", False))
            patched.append(torch.utils.data.DataLoader(
                dl.dataset, batch_size=self.config.loader.batch_size,
                num_workers=self.config.loader.num_workers,
                pin_memory=self.config.loader.pin_memory,
                sampler=sampler, shuffle=False,
                persistent_workers=self.config.loader.persistent_workers))
        self.trainer.fit_loop._combined_loader.flattened = patched

    # ----------------- optimiser & sched -----------------------------
    def configure_optimizers(self):
        optim_args = {
            'lr': self.config.optim.lr,
            'betas': (self.config.optim.beta1, self.config.optim.beta2),
            'eps': self.config.optim.eps,
            'weight_decay': self.config.optim.weight_decay,
        }

        optimizer = torch.optim.AdamW(
            itertools.chain(self.ratio_model.parameters(), self.noise.parameters()),
            **optim_args
        )

        scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]