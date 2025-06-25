import os

import fsspec
import hydra
import lightning as L
import omegaconf
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


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
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

def train_ratio_model(config, train_ds_src, valid_ds_src,callbacks,
                        train_ds_tgt, valid_ds_tgt, wandb_logger, ckpt_path):
    """Train the ratio model."""
    logger = utils.get_logger(__name__)
    logger.info('Starting Ratio Model Training.')

    # make check that the configuration and loader are compatible
    _ = utils.make_checks_if_config_and_loader_is_synchronized(
        config, train_ds_src, valid_ds_src, train_ds_tgt, valid_ds_tgt)
    # Todo: add the pretrained backbone loading
    train_loader = dataloader.build_mixed_loader(train_ds_src, train_ds_tgt, config)
    valid_loader = dataloader.build_mixed_loader(valid_ds_src, valid_ds_tgt, config)


    # ------- 1) time-independent classifier -------
    cls_ti, loaded_ti = utils.build_or_load(
        classifier.Classifier,
        dict(config=config, tokenizer=train_ds_src.tokenizer,
             train_time_independent=True),
        config.classifier_ti.ckpt_path,
        config.classifier_ti.retrain_when_loaded,
    )
    if (not loaded_ti) or config.classifier_ti.retrain_when_loaded:
        trainer = hydra.utils.instantiate(  # unchanged
            config.trainer_ti,
            default_root_dir=os.path.join(
                config.checkpointing.save_dir, "ti_classifier"),
            callbacks=utils._callbacks_for("ti_classifier", config.checkpointing.save_dir,
                                           callbacks, config.training_classifier.val_metric_for_best_model),
            strategy=hydra.utils.instantiate(config.strategy),
            logger=wandb_logger,
        )
        trainer.fit(cls_ti, train_loader, valid_loader,
                    ckpt_path=config.classifier_ti.ckpt_path or None)

    # ------- 2) time-dependent classifier -------
    cls_td, loaded_td = utils.build_or_load(
        classifier.Classifier,
        dict(config=config, tokenizer=train_ds_src.tokenizer,
             train_time_independent=False),
        config.classifier_td.ckpt_path,
        config.classifier_td.retrain_when_loaded,
    )
    if (not loaded_td) or config.classifier_td.retrain_when_loaded:
        trainer = hydra.utils.instantiate(  # unchanged
            config.trainer_td,
            default_root_dir=os.path.join(
                config.checkpointing.save_dir, "td_classifier"),
            callbacks=utils._callbacks_for("td_classifier", config.checkpointing.save_dir, callbacks,
                                           config.training_classifier.val_metric_for_best_model),
            strategy=hydra.utils.instantiate(config.strategy),
            logger=wandb_logger,
        )
        trainer.fit(cls_td, train_loader, valid_loader,
                    ckpt_path=config.classifier_td.ckpt_path or None)

    # ------- 3) ratio model -------
    train_paired_loader = L.pytorch.utilities.combined_loader.CombinedLoader(
        {"src": train_ds_src, "tgt": train_ds_tgt}, mode="max_size_cycle")
    valid_paired_loader = L.pytorch.utilities.combined_loader.CombinedLoader(
        {"src": valid_ds_src, "tgt": train_ds_tgt}, mode="max_size_cycle")
    ratio_net, loaded_ratio = utils.build_or_load(
        ratio.RatioEstimator,
        dict(
            config=config,
            tokenizer=train_ds_src.tokenizer,
            domain_classifier=cls_ti,
            domain_classifier_time_dependent=cls_td,
        ),
        config.ratio_model.ckpt_path,
        config.ratio_model.retrain_when_loaded,
    )
    if (not loaded_ratio) or config.ratio_model.retrain_when_loaded:
        trainer = hydra.utils.instantiate(
            config.trainer_ratio,
            default_root_dir=os.path.join(
                config.checkpointing.save_dir, "ratio_model"),
            callbacks=utils._callbacks_for("ratio_model", config.checkpointing.save_dir, callbacks,
                                           config.training_ratio.val_metric_for_best_model),
            strategy=hydra.utils.instantiate(config.strategy),
            logger=wandb_logger,
        )
        trainer.fit(ratio_net, train_paired_loader, valid_paired_loader,
                    ckpt_path=config.ratio_model.ckpt_path or None)
def _train(config, logger, tokenizer, train_ratio=False):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    # Prepare wandb logger keyword arguments
    wandb_kwargs = {
      'config': omegaconf.OmegaConf.to_object(config),
      **config.wandb}

    # If training the ratio model, append '-ratio' to the run name
    if train_ratio and 'name' in wandb_kwargs:
        wandb_kwargs['name'] = f"{wandb_kwargs['name']}-ratio"
    wandb_logger = L.pytorch.loggers.WandbLogger(**wandb_kwargs)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer, domain='src')
  #_print_batch(train_ds, valid_ds, tokenizer)

  if train_ratio:
    # For training the ratio model both source and target datasets are needed
    train_ds_tgt, valid_ds_tgt = dataloader.get_dataloaders(config, tokenizer, domain='tgt')
    train_ratio_model(config=config,
                      train_ds_src=train_ds, valid_ds_src=valid_ds,
                      callbacks=callbacks,
                      train_ds_tgt=train_ds_tgt, valid_ds_tgt=valid_ds_tgt,
                      wandb_logger=wandb_logger, ckpt_path=ckpt_path)
  else:
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
      config.trainer,
      default_root_dir=os.getcwd(),
      callbacks=callbacks,
      strategy=hydra.utils.instantiate(config.strategy),
      logger=wandb_logger)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer, train_ratio = 'ratio' in config.mode)


if __name__ == '__main__':
  main()