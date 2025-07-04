import os
import typing

import fsspec
import hydra
import lightning as L
import omegaconf
from omegaconf import DictConfig
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils
import classifier
import ratio

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
if torch.cuda.is_available():
  omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
else:
    omegaconf.OmegaConf.register_new_resolver(
    'device_count', lambda: 1)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up',  lambda x, y: (x + y - 1) // y if y else x)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.

  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def _get_resume_ckpt(cfg: DictConfig) -> typing.Union[str, None]:
    """Return a valid resume checkpoint path if it exists."""
    if (
        cfg.checkpointing.resume_from_ckpt
        and cfg.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(cfg.checkpointing.resume_ckpt_path)
    ):
        return cfg.checkpointing.resume_ckpt_path
    return None


def _make_trainer(cfg: DictConfig, callbacks: list, wandb_logger):
    """Instantiate a Lightning trainer from a Hydra config section."""
    return hydra.utils.instantiate(
        cfg.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(cfg.get("strategy")),
        logger=wandb_logger,
    )


# -----------------------------------------------------------------------------
# Diffusion training
# -----------------------------------------------------------------------------
def train_diffusion_model(cfg: DictConfig, callbacks: list, wandb_logger, tokenizer):
    """Train the diffusion model only."""
    train_ds, valid_ds = dataloader.get_dataloaders(cfg, tokenizer, domain="src")

    model = diffusion.Diffusion(cfg, tokenizer=valid_ds.tokenizer)
    trainer = _make_trainer(cfg, callbacks, wandb_logger)

    trainer.fit(model, train_ds, valid_ds, ckpt_path=_get_resume_ckpt(cfg))


# -----------------------------------------------------------------------------
# Classifier helpers
# -----------------------------------------------------------------------------
def _build_and_maybe_train_classifier(*, cfg: DictConfig, tokenizer, train_time_independent: bool,
                                      loader_pair: tuple, section_cfg: DictConfig,
                                      trainer_cfg: DictConfig, name: str,
                                      callbacks: list, wandb_logger):
    """Create (or load) a classifier and train if necessary."""
    cls, loaded = utils.build_or_load(
        classifier.Classifier,
        dict(
            config=cfg,
            tokenizer=tokenizer,
            train_time_independent=train_time_independent,
        ),
        section_cfg.ckpt_path,
        section_cfg.retrain_when_loaded,
    )

    if (not loaded) or section_cfg.retrain_when_loaded:
        train_loader, valid_loader = loader_pair
        trainer = _make_trainer(cfg, utils._callbacks_for(name, cfg.checkpointing.save_dir, callbacks, cfg.training_classifier.val_metric_for_best_model), wandb_logger)
        trainer.fit(cls, train_loader, valid_loader, ckpt_path=section_cfg.ckpt_path or None)
    return cls


def train_classifier_models(cfg: DictConfig, callbacks: list, wandb_logger, tokenizer, train_loader, valid_loader):
    """Train / load both time‑independent and time‑dependent classifiers."""
    cls_ti = _build_and_maybe_train_classifier(
        cfg=cfg,
        tokenizer=tokenizer,
        train_time_independent=True,
        loader_pair=(train_loader, valid_loader),
        section_cfg=cfg.classifier_ti,
        trainer_cfg=cfg.trainer_ti,
        name="ti_classifier",
        callbacks=callbacks,
        wandb_logger=wandb_logger,
    )
    cls_td = _build_and_maybe_train_classifier(
        cfg=cfg,
        tokenizer=tokenizer,
        train_time_independent=False,
        loader_pair=(train_loader, valid_loader),
        section_cfg=cfg.classifier_td,
        trainer_cfg=cfg.trainer_td,
        name="td_classifier",
        callbacks=callbacks,
        wandb_logger=wandb_logger,
    )
    return cls_ti, cls_td


# -----------------------------------------------------------------------------
# Ratio training pipeline
# -----------------------------------------------------------------------------

def train_ratio_pipeline(cfg: DictConfig, callbacks: list, wandb_logger, tokenizer):
    """Full pipeline for training source/target classifiers and the ratio model."""

    # 1. Build mixed loaders for source and target domains
    train_ds_src, valid_ds_src = dataloader.get_dataloaders(cfg, tokenizer, domain="src")
    train_ds_tgt, valid_ds_tgt = dataloader.get_dataloaders(cfg, tokenizer, domain="tgt")

    train_loader = dataloader.build_mixed_loader(train_ds_src, train_ds_tgt, cfg)
    valid_loader = dataloader.build_mixed_loader(valid_ds_src, valid_ds_tgt, cfg)

    # 2. Train / load classifiers
    cls_ti, cls_td = train_classifier_models(
        cfg, callbacks, wandb_logger, tokenizer, train_loader, valid_loader
    )

    # 3. Train / load ratio estimator
    paired_train = L.pytorch.utilities.combined_loader.CombinedLoader(
        {"src": train_ds_src, "tgt": train_ds_tgt}, mode="max_size_cycle"
    )
    paired_valid = L.pytorch.utilities.combined_loader.CombinedLoader(
        {"src": valid_ds_src, "tgt": valid_ds_tgt}, mode="max_size_cycle"
    )

    ratio_net, loaded_ratio = utils.build_or_load(
        ratio.RatioEstimator,
        dict(
            config=cfg,
            tokenizer=train_ds_src.tokenizer,
            domain_classifier=cls_ti,
            domain_classifier_time_dependent=cls_td,
        ),
        cfg.ratio_model.ckpt_path,
        cfg.ratio_model.retrain_when_loaded,
    )

    if (not loaded_ratio) or cfg.ratio_model.retrain_when_loaded:
        trainer = _make_trainer(
            cfg,
            utils._callbacks_for("ratio_model", cfg.checkpointing.save_dir, callbacks, cfg.training_ratio.val_metric_for_best_model),
            wandb_logger,
        )
        trainer.fit(ratio_net, paired_train, paired_valid, ckpt_path=cfg.ratio_model.ckpt_path or None)


# -----------------------------------------------------------------------------
# Evaluation helpers – unchanged from the original script
# -----------------------------------------------------------------------------
def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]

    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples



def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Dispatch to the appropriate training or evaluation routine."""
    L.seed_everything(cfg.seed)
    _print_config(cfg, resolve=True, save_cfg=True)
     # e.g. 1024 or 512

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(cfg)

    # Setup logging
    wandb_logger = None
    if cfg.get("wandb") is not None:
        wandb_kwargs = {
            "config": omegaconf.OmegaConf.to_object(cfg),
            **cfg.wandb,
        }
        wandb_logger = L.pytorch.loggers.WandbLogger(**wandb_kwargs)

    callbacks = []
    if "callbacks" in cfg:
        callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks.values()]

    # Dispatch by mode

    if cfg.mode == "train":
        train_diffusion_model(cfg, callbacks, wandb_logger, tokenizer)
    elif cfg.mode == "train_ratio":
        train_ratio_pipeline(cfg, callbacks, wandb_logger, tokenizer)
    elif cfg.mode ==  "sample_eval":
        generate_samples(cfg, logger, tokenizer)
    elif cfg.mode == "ppl_eval":
        _ppl_eval(cfg, logger, tokenizer)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")



if __name__ == '__main__':
  main()