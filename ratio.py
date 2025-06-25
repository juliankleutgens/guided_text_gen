import itertools, typing, torch, torch.nn as nn, torchmetrics
import lightning as L
import hydra.utils
import dataloader, noise_schedule, utils        # unchanged imports
import models
from base_dm_model import BaseDMModel
# -------------------------------------------------------------------

class RatioEstimator(BaseDMModel):
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
        # --------------------------------------------------
        # 0 ▸ Minimal attributes required by BaseDMModel
        self.config = config                # Hydra cfg node
        super().__init__()                  # gives self.dtype

        # --------------------------------------------------
        # 1 ▸ High‑level flags & helpers
        self.tokenizer = tokenizer
        self.train_time_independent = False   # ratio net is always t‑conditioned

        # --------------------------------------------------
        # 2 ▸ Vocabulary & special tokens
        self.vocab_size = tokenizer.vocab_size
        if getattr(tokenizer, "mask_token", None) is None:
            self.mask_index = self.vocab_size
            self.vocab_size += 1             # add [MASK]
        else:
            self.mask_index = tokenizer.mask_token_id

        # --------------------------------------------------
        # 3 ▸ Training‑ratio hyper‑parameters
        tr_cfg = config.training_ratio
        self.antithetic_sampling = tr_cfg.antithetic_sampling
        self.importance_sampling = tr_cfg.importance_sampling
        self.change_of_variables = tr_cfg.change_of_variables
        self.eta1, self.eta2 = tr_cfg.eta1, tr_cfg.eta2
        self.sampling_eps       = tr_cfg.sampling_eps
        self.consistency        = bool(self.eta2 and self.eta2 > 0)
        self.cycle              = bool(self.eta1 and self.eta1 > 0)

        self.diffusion = config.diffusion
        self.time_conditioning = config.time_conditioning
        self.T         = config.T

        # --------------------------------------------------
        # 4 ▸ Frozen auxiliary classifiers
        self.domain_classifier = domain_classifier.eval().requires_grad_(False)
        self.domain_classifier_t = (
            domain_classifier_time_dependent.eval().requires_grad_(False)
        )

        # --------------------------------------------------
        # 5 ▸ Ratio‑network backbone
        if config.ratio_backbone == "dit":
            self.ratio_model = models.dit.DITRatio(
                config, vocab_size=self.vocab_size, time_conditioning=True
            )
        else:
            raise NotImplementedError(
                f"Ratio backbone '{config.ratio_backbone}' not implemented."
            )
        if pretrained_backbone is not None:
            self.ratio_model.load_pretrained_encoder(pretrained_backbone)

        utils.print_num_parameters(self.ratio_model, print_prefix="Ratio model ")

        # --------------------------------------------------
        # 6 ▸ Noise schedule  (needs self.dtype)
        self.noise = noise_schedule.get_noise(config, dtype=self.dtype)

        # --------------------------------------------------
        # 7 ▸ Metrics & loss
        self.mse = nn.MSELoss()
        base_metrics = torchmetrics.MetricCollection({"loss": torchmetrics.MeanMetric()})
        self.train_metrics = base_metrics.clone(prefix="train/")
        self.valid_metrics = base_metrics.clone(prefix="val/")

        # --------------------------------------------------
        # 8 ▸ Fast‑forward placeholders
        self.fast_forward_epochs  = None
        self.fast_forward_batches = None

    # ================================================================
    # ▸ forward interface
    def forward(self, x_t, sigma, attention_mask=None, x_emb=None):
        # x_t: (B, L) int tokens     t: (B,) float in (0,1]
        sigma = self._process_sigma(sigma) if sigma is not None else sigma
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.ratio_model(x_t, sigma, x_emb=x_emb, attention_mask=attention_mask)
        return logits

    # ================================================================
    # ▸ core loss computation
    def _compute_losses(self, batch):
        batch_src = batch["src"]  # was: batch
        batch_tgt = batch["tgt"]

        x0_src, x0_tgt = batch_src["input_ids"], batch_tgt["input_ids"]
        attention_mask_src = batch_src["attention_mask"]
        attention_mask_tgt = batch_tgt["attention_mask"]
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
        r_pred_src = self.forward(x_t_src, sigma=sigma_src, attention_mask=attention_mask_src)
        loss_ratio = self.mse(r_pred_src, r_src)

        # ---------- cycle loss on target branch ----------
        if self.cycle:
            B_tgt = x0_tgt.size(0)
            t_tgt = self._sample_t(B_tgt)
            sigma_tgt, _ = self.noise(t_tgt)
            move_chance_tgt = 1 - torch.exp(-sigma_tgt)
            x_t_tgt = self._q_xt(x0_tgt, move_chance_tgt[:, None])
            with torch.no_grad():
                c_tdep = self.domain_classifier_t(x_t_tgt, sigma_tgt, attention_mask=attention_mask_tgt).squeeze(-1)
                r_tdep = (-c_tdep if not self.config.training_ratio.classifier_output_with_sigmoid
                          else torch.log((1 - c_tdep) / (c_tdep + 1e-8) + 1e-8))
            r_pred_tgt = self(x_t_tgt, sigma=sigma_tgt, attention_mask=attention_mask_tgt)
            loss_cycle = self.mse(r_pred_tgt, r_tdep)
        else:
            loss_cycle = torch.zeros((), device=self.device)

        total = loss_ratio + self.eta1 * loss_cycle
        return total, loss_ratio, loss_cycle

    # ================================================================
    # ▸ Lightning overrides
    def training_step(self, batch, _):
        total, l_r, l_cy = self._compute_losses(batch)
        self.train_metrics.update(torch.tensor([total.detach()]))

        self.log_dict({
            "train/total": total,
            "train/L_ratio": l_r,
            "train/L_cycle": l_cy,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }, prog_bar=True, sync_dist=True)
        return total

    def validation_step(self, batch, _):
        total, l_r, l_cy = self._compute_losses(batch)
        self.valid_metrics.update(torch.tensor([total.detach()]))
        self.log_dict({
            "val/total": total,
            "val/L_ratio": l_r,
            "val/L_cycle": l_cy,
        }, prog_bar=True, sync_dist=True)


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