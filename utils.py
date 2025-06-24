"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import logging
import math
import os
import fsspec
import lightning
import torch
from timm.scheduler import CosineLRScheduler


def filter_arxiv_dataset_by_domain(dataset, domain_key):
  """
  Filter an arXiv metadata Dataset to only include records whose
  *primary* category prefix matches domain_key.
  """
  if not domain_key:
    return dataset

  def _matches_primary(example):
    cats = example.get("categories")
    # get list of tags
    tags = cats.split() if isinstance(cats, str) else cats if isinstance(cats, list) else []
    if not tags:
      return False
    # only inspect the first (primary) tag
    primary = tags[0]
    prefix = primary.split('.', 1)[0]
    return prefix == domain_key

  return dataset.filter(_matches_primary)


def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()

# ── helpers ─────────────────────────────────────────────────────────────
def _unwrap_dataset(ds):
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds

def _infer_wrap_from_filename(loader):
    """
    Return True / False if the cached filename tells us,
    otherwise return None (meaning “unknown”).
    """
    ds = _unwrap_dataset(loader.dataset)
    if getattr(ds, "cache_files", None):
        fname = ds.cache_files[0]["filename"]
        if "_wrapped.dat"   in fname: return True
        if "_unwrapped.dat" in fname: return False
    return None   # no hint found
# ── main check ──────────────────────────────────────────────────────────
def make_checks_if_config_and_loader_is_synchronized(config, *dataloaders):
    loaders = [dl for dl in dataloaders if dl is not None]
    if not loaders:
        raise ValueError("No dataloaders provided.")

    # 1 ▸ tokenizer / model_max_length consistency
    ref_tok = loaders[0].tokenizer
    for i, dl in enumerate(loaders[1:], 1):
        tok = dl.tokenizer
        if tok.name_or_path   != ref_tok.name_or_path \
        or tok.vocab_size     != ref_tok.vocab_size \
        or tok.model_max_length != ref_tok.model_max_length:
            raise ValueError(f"Tokenizer mismatch between loader[0] and loader[{i}].")

    # 2 ▸ wrap check (filename-only, no data access)
    inferred = _infer_wrap_from_filename(loaders[0])
    if inferred is not None and inferred != config.data.wrap:
        raise ValueError(f"Config says wrap={config.data.wrap} but cache looks "
                         f"{'wrapped' if inferred else 'unwrapped'}.")

    # 3 ▸ simple batch-size sanity
    if getattr(config.loader, "batch_size", 1) <= 0:
        raise ValueError("Batch size must be positive.")

    return {
        "wrap"          : inferred if inferred is not None else "unknown",
        "tokenizer"     : ref_tok.name_or_path,
        "model_max_len" : ref_tok.model_max_length,
    }


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


class Sampler:
  def __init__(self, shape):
    self.shape = shape

  def _sampling_noise(self):
    pass
  
  def _hard_sample(self, logits):
    pass

  def _soft_sample(self, logits):
    return 0

  def sample(self, logits):
    noise = self._sampling_noise()
    noise = noise[: logits.shape[0], :]
    logits = logits + noise.to(
      dtype=logits.dtype, device=logits.device)
    hard_sample = self._hard_sample(logits)
    soft_sample = self._soft_sample(logits)
    return soft_sample + (hard_sample - soft_sample).detach()


class TopKSampler(Sampler):
  def __init__(self, k, shape, gamma_tau=1.0):
    super().__init__(shape)
    self.k = k
    self.gamma_tau = gamma_tau
    self.num_betas = 10
    self.sampler = torch.distributions.gamma.Gamma(
      1 / k * torch.ones(self.num_betas, * self.shape), 1.0)

  def _sampling_noise(self):
    noise = self.sampler.sample()
    beta = self.k / torch.arange(1, self.num_betas + 1, 1,
                                 dtype=torch.float32)
    beta = beta[:, None, None]
    assert beta.ndim == noise.ndim
    s = noise / beta
    s = torch.sum(s, axis=0)
    s = s - math.log(10.0)
    s = self.gamma_tau * (s / self.k)
    return s

  def _hard_sample(self, logits):
    assert logits.ndim == 2
    thresholds, _ = torch.sort(logits, dim=-1)
    thresholds = thresholds[:, - self.k][:, None]
    return (logits >= thresholds).type(logits.dtype)

  def _soft_sample(self, logits):
    soft_top_k = logits - torch.mean(logits, dim=-1,
                                     keepdim=True)
    return soft_top_k / torch.norm(soft_top_k, dim=-1,
                                   keepdim=True)


class DeterministicTopK(TopKSampler):
  def __init__(self, k):
    super().__init__(k, shape=(1, 1))

  def _sampling_noise(self):
    return 0

  def discreize(self, x):
    hard_sample = self._hard_sample(x)
    soft_sample = self._soft_sample(x)
    return soft_sample + (hard_sample - soft_sample).detach()

class GumbelSampler(Sampler):

  def __init__(self, shape, temperature=1.0):
    super().__init__(shape)
    self.temperature = temperature

  def _sampling_noise(self):
    return - (1e-10 - (
      torch.rand(* self.shape) + 1e-10).log()).log()

  def _hard_sample(self, logits):
    assert logits.ndim == 2
    indices = torch.argmax(logits, dim=-1)
    zeros = logits * 0
    ones = torch.ones_like(logits[:, :, :1])
    return torch.scatter(zeros, -1, indices[:, :, None],
                         ones)

  def _soft_sample(self, logits):
    return torch.nn.functional.softmax(
      logits / self.temperature, dim=-1)


class BinarySampler(GumbelSampler):

  def sample(self, probs):
    # TODO(subhamsahoo): use the temperature parameter.
    pos_noise = self._sampling_noise().to(
      dtype=probs.dtype, device=probs.device)
    neg_noise = self._sampling_noise().to(
      dtype=probs.dtype, device=probs.device)
    del_noise_exp = (neg_noise - pos_noise).exp()
    hard_sample = (probs * (1 + del_noise_exp)
                   > 1).to(probs.dtype)
    soft_sample = probs / (probs + (1 - probs) * del_noise_exp)
    return soft_sample + (hard_sample - soft_sample).detach()


class GaussianSampler:
  def __init__(self):
    self.softplus = torch.nn.Softplus()

  def sample(self, x):
    assert x.ndim == 2
    n = x.shape[-1] // 2
    mu = x[:, :n]
    sigma = self.softplus(x[:, n:]).sqrt()
    return mu + sigma * torch.randn_like(mu)



@lightning.pytorch.utilities.rank_zero_only
def print_num_parameters(model: torch.nn.Module, verbose: bool = True, print_prefix="") -> None:
    """
    Prints the total and trainable number of parameters in a model.

    Args:
        model: the torch.nn.Module whose parameters to count.
        verbose: if True, prints the counts; otherwise returns nothing.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"{print_prefix}Total parameters: {total_params:,}")
        print(f"{print_prefix}Trainable parameters: {trainable_params:,}")


def build_or_load(model_cls, init_kwargs, ckpt_path=None, freeze=False):
    if ckpt_path and os.path.exists(ckpt_path):
        model = model_cls.load_from_checkpoint(ckpt_path, **init_kwargs)
        loaded = True
    else:
        model = model_cls(**init_kwargs)
        loaded = False

    # --- make sure the checkpoint really is the right kind of model -----------
    if loaded and not isinstance(model, model_cls):
        raise TypeError(
            f"Checkpoint at '{ckpt_path}' contains a "
            f"{type(model).__name__}; expected {model_cls.__name__}"
        )


    return model, loaded



    # --- helper: clone callbacks and route checkpoints ----------
import copy

def _callbacks_for(subdir: str, checkpoint_save_dir: str, callbacks):
    """Return a deep‑copied callbacks list whose ModelCheckpoint
    saves to .../<subdir>/checkpoints/.
    #config.checkpointing.save_dir
    """
    cbs = []
    for cb in callbacks:
        cb_new = copy.deepcopy(cb)
        if isinstance(cb_new, lightning.pytorch.callbacks.ModelCheckpoint):
            cb_new.dirpath = os.path.join(
                checkpoint_save_dir, subdir, "checkpoints"
            )
            # Special‑case: for the ratio model we want to track a different metric
            # ToDo: this is a hack, should be fixed in the config make a new config for ratio model
            if subdir == 'ratio_model' and getattr(cb_new, 'monitor', None) == 'val/cross_entropy':
                cb_new.monitor = 'val/total'  # RatioEstimator logs val/total, not val/cross_entropy
        cbs.append(cb_new)
    return cbs

