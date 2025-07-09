import itertools
import typing

import hydra.utils
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import transformers

import dataloader
import models.dit
import noise_schedule
import utils
from base_dm_model import BaseDMModel

class MicroAveragingMetric(torchmetrics.Metric):
  """Micro-averaging metric.

    Adapted from https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py#L12
  """

  def __init__(self, class_idx: typing.Optional[int] = 1,
               dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.class_idx = torch.tensor(class_idx) \
      if class_idx is not None else None
    self.add_state("numerator", default=torch.tensor(0.0),
                   dist_reduce_fx="sum")
    self.add_state("denominator", default=torch.tensor(0.0),
                   dist_reduce_fx="sum")

  def _update(
      self, numerator, denominator, preds, y) -> tuple:
    raise NotImplementedError

  def update(self, logits: torch.Tensor, y: torch.Tensor):
    # Support both multi‑class (C≥2) and single‑logit binary classifiers.
    if logits.size(-1) == 1:
      preds = (logits > 0).long().squeeze(-1)   # 0/1 predictions
    else:
      preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    assert preds.shape == y.shape, \
      f"preds shape {preds.shape} != y shape {y.shape}"
    self.numerator, self.denominator = self._update(
      self.numerator, self.denominator, preds, y)

  def compute(self):
    # compute final result
    value = self.numerator.float() / self.denominator \
      if self.denominator.item() > 0. else torch.tensor(0.0)
    return value

  def reset(self):
    self.numerator = torch.tensor(0.0).to(self.device)
    self.denominator = torch.tensor(0.0).to(self.device)


class CrossEntropy(MicroAveragingMetric):
  """Calculates cross-entropy loss."""
  def _update(
      self, numerator, denominator, logits, y) -> tuple:
    with torch.no_grad():
      if logits.size(-1) == 1:           # single‑logit → BCE
        numerator += F.binary_cross_entropy_with_logits(
          logits.view(-1), y.float(), reduction='sum'
        )
      else:                              # multi‑class → CE
        numerator += F.cross_entropy(
          logits.view(-1, logits.size(-1)),
          y.view(-1),
          ignore_index=-100,
          reduction='sum'
        )
      denominator += y.numel()
    return numerator, denominator

  # Overrides parent class to use logits and not (argmax) preds
  def update(self, logits: torch.Tensor, y: torch.Tensor):
    y = y.view(-1)
    self.numerator, self.denominator = self._update(
      self.numerator, self.denominator, logits, y)


class Accuracy(MicroAveragingMetric):
  """Calculates accuracy.

    Can be used to calculate accuracy per class.
    Copied from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(
      self, numerator, denominator, preds, y) -> tuple:
    if self.class_idx is None:
      numerator += (preds == y).sum()
      denominator += y.numel()
    else:
      class_idx = self.class_idx
      relevant_idxs = (y == class_idx)
      numerator += (preds[relevant_idxs] == class_idx).sum()
      denominator += relevant_idxs.sum()
      relevant_idxs = (y != class_idx)
      numerator += (preds[relevant_idxs] != class_idx).sum()
      denominator += relevant_idxs.sum()
    return numerator, denominator


class Precision(MicroAveragingMetric):
  """Calculates precision.

    Can be used to calculate precision per class.
    Adapted from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(self, numerator, denominator, preds, y) -> tuple:
    class_idx = self.class_idx
    relevant_idxs = (preds == class_idx)
    numerator += (y[relevant_idxs] == class_idx).sum()
    denominator += relevant_idxs.sum()
    return numerator, denominator


class Recall(MicroAveragingMetric):
  """Calculate recall.

    Can be used to calculate recall per class.
    Adapted from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(self, numerator, denominator, preds, y) -> tuple:
    class_idx = self.class_idx
    relevant_idxs = (y == class_idx)
    numerator += (preds[relevant_idxs] == class_idx).sum()
    denominator += relevant_idxs.sum()
    return numerator, denominator


class Classifier(BaseDMModel):
  def __init__(
        self,
        config,
        tokenizer: transformers.PreTrainedTokenizer,
        pretrained_backbone: typing.Optional[torch.nn.Module] = None,
        train_time_independent: bool = False,
    ):
      # --------------------------------------------------
      # 0 ▸ Minimal attributes required by BaseDMModel
      self.config = config
      super().__init__()  # <- gives us self.dtype

      # --------------------------------------------------
      # 1 ▸ High‑level flags
      self.train_time_independent = train_time_independent
      self.is_eval_classifier = getattr(config, "is_eval_classifier", False)

      # --------------------------------------------------
      # 2 ▸ Tokenizer & vocabulary
      self.tokenizer = tokenizer
      self.vocab_size = tokenizer.vocab_size
      if not getattr(tokenizer, "mask_token", None):
          self.mask_index = self.vocab_size
          self.vocab_size += 1
      else:
          self.mask_index = tokenizer.mask_token_id

      # --------------------------------------------------
      # 3 ▸ Training‑schedule hyper‑parameters
      tr_cfg = config.training_classifier
      self.antithetic_sampling = tr_cfg.antithetic_sampling
      self.importance_sampling = tr_cfg.importance_sampling
      self.change_of_variables = tr_cfg.change_of_variables
      self.sampling_eps = tr_cfg.sampling_eps

      self.T = config.T
      self.lr = config.optim.lr
      self.time_conditioning = config.time_conditioning

      # --------------------------------------------------
      # 4 ▸ Noise schedule (needs self.dtype from parent)
      self.noise = noise_schedule.get_noise(config, dtype=self.dtype)

      # --------------------------------------------------
      # 5 ▸ Backbone construction
      if config.classifier_backbone == "dit":
          self.classifier_model = models.dit.DITClassifier(
              config,
              vocab_size=self.vocab_size,
              time_conditioning=not train_time_independent,
          )
          """
          for name, module in self.classifier_model.named_modules():
            if isinstance(module, torch.nn.Dropout):
              print(f"{name}: p={module.p}")
          """
      else:
          raise NotImplementedError(
              f"Classifier backbone {config.classifier_backbone} not implemented."
          )

      if pretrained_backbone is not None:
          # For PPLM / NoS fine‑tuning
          self.classifier_model.load_pretrained_encoder(pretrained_backbone)

      #utils.print_num_parameters(self.classifier_model, print_prefix="Classifier model ")

      # --------------------------------------------------
      # 6 ▸ Metrics
      metrics = torchmetrics.MetricCollection(
          {
              "cross_entropy": CrossEntropy(),
              "accuracy": Accuracy(class_idx=None),
              "precision": Precision(class_idx=1),
              "recall": Recall(class_idx=1),
          }
      )
      metrics.set_dtype(torch.float64)
      self.train_metrics = metrics.clone(prefix="train/")
      self.valid_metrics = metrics.clone(prefix="val/")

      # --------------------------------------------------
      # 7 ▸ Fast‑forward placeholders (set during training)
      self.fast_forward_epochs = None
      self.fast_forward_batches = None

  def forward(self, x, sigma=None, x_emb=None, attention_mask=None):
    """Returns logits.

      x_emb can be provided during PPLM / NoS-style guidance
      (see: https://arxiv.org/abs/2305.20009).
    """
    if self.is_eval_classifier:
      logits = self.classifier_model(x)
      if hasattr(logits, 'logits'):
        logits = logits.logits
    else:
      sigma = self._process_sigma(sigma) if sigma is not None or not self.train_time_independent else sigma
      with torch.cuda.amp.autocast(dtype=torch.float32):
        logits = self.classifier_model(x, sigma, x_emb=x_emb, attention_mask=attention_mask)
    return logits

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True)
    self.log(name='lr',
             value=
             self.trainer.optimizers[0].param_groups[0][
               'lr'],
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True, logger=False)
    return loss

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      itertools.chain(self.classifier_model.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  def _compute_loss(self, batch, prefix):
    # get data
    x0 = batch['input_ids']
    y = batch['label']
    attention_mask = batch.get('attention_mask', None)
    t = None

    # get the output logits
    if self.is_eval_classifier:
      logits = self.forward(x0)
    elif not self.train_time_independent:
        t = self._sample_t(x0.size(0))
        time_cond, move_chance = self._get_time_conditioning_and_move_chance(t)
        x_in = self._q_xt(x0, move_chance)
        logits = self.forward(x_in, time_cond, attention_mask=attention_mask)
    else:                     # clean input for val / test in time dependent case
        logits = self.forward(x0, attention_mask=attention_mask)


    # Optional label‑smoothing handled by PyTorch (for multi‑class) or manual (for single‑logit binary)
    targets = y.float()
    if (not self.is_eval_classifier
        and getattr(self.config.training_classifier, 'use_label_smoothing', False)
        and prefix == 'train'):
      mu   = getattr(self.config.training_classifier, 'label_smoothing_eps', 0.1)
      var  = 0.005   # e.g. (std=0.1)**2; adjust as you like
      temp = mu * (1 - mu) / var - 1
      alpha = mu * temp
      beta  = (1 - mu) * temp
      dist = torch.distributions.Beta(alpha, beta)
      eps = dist.sample(targets.shape).to(targets.device)
      targets = targets * (1.0 - eps) + (1.0 - targets) * eps

    # I have deleted the FUDGE implementation, since it is only useful for autoregressive models (https://arxiv.org/pdf/2104.05218)

    logits_flat = logits.view(-1)
      # BCE expects float targets
    loss = F.binary_cross_entropy_with_logits(
        logits_flat,
        targets,
        reduction='mean'
    )

    if prefix == 'train':
      self.train_metrics.update(logits, y)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(logits, y)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(logits, y)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss
  
  def detokenize_batch(x0: torch.Tensor, tokenizer, *, skip_special_tokens: bool = True):
    """
    How to use:
    texts = self.detokenize_batch(x0, self.tokenizer)

    for i, txt in enumerate(texts[:3]):   # show first few examples
      print(f"[sample {i}] {txt}")
    """

    input_ids = x0.detach().cpu().tolist()
    return tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)