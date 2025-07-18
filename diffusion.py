import itertools
import math
import os
from tqdm import tqdm
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import ratio
import noise_schedule
import utils
from base_dm_model import BaseDMModel

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(BaseDMModel):
  # ==========================================
  # 0. Initialization
  # ==========================================
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    self.config = config
    super().__init__()
    self.save_hyperparameters()

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.inference_mode = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()
    self.diffusion = config.diffusion

    # sample congiguration
    self.validation_mode = config.mode == "ppl_eval"
    self.guided_sampling = self.config.sampling.guided_sampling
    self.guidance_scale = self.config.sampling.guidance_scale
    if self.guided_sampling:
      self.ratio_model, _ = utils.build_or_load(
        model_=ratio.RatioEstimator,
        init_kwargs=dict(
            config=config,
            tokenizer=tokenizer,
            domain_classifier=None,
            domain_classifier_time_dependent=None,
            inference_mode=True
        ),
        ckpt_path=config.ratio_model.ckpt_path,
        freeze=True,)
      self.ratio_model.eval()           # inference only
    self._ratio_flat_num = 0  # int
    self._ratio_flat_den = 0  # int
    self.TOPK_LIST = (100, 500, 1000, 2000, 5000, 10000, 15000, 20000)
    self._topk_vs_t    = {k: [] for k in self.TOPK_LIST}  # diffusion vs ratio
    self._topk_dg_vs_t = {k: [] for k in self.TOPK_LIST}  # diffusion vs guided

    self._t_records = []  # float t per (informative) batch


  # ==========================================
  # 1. Configuration validation and checkpointing
  # ==========================================
  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def optimizer_step(self, *args, **kwargs):
    """The optimizer step for the Hydra."""
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  # ==========================================
  # 3. Loss computation and parameterization for three types
  #    of parameterization: subs, d3pm, sedd
  # ==========================================
  def _subs_parameterization(self, logits, xt):
    # "Zero Masking Prob":
    # log prob at the mask index = - infinity
    logits[..., self.mask_index] += self.neg_infinity

    # "Copy over":
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0

    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    return logits.log_softmax(dim=-1)

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)
    # The below scatter operation sets the log score for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],torch.zeros_like(logits[..., :1]))
    return logits

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma) # if time_conditioning==False, then sigma=zeros 
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    return logits

  # ==========================================
  # 4. Loss computation
  # ==========================================
  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    #x_clean_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    #perplexity_of_clean_text = self.get_generative_perplexity(x_clean_text, retokenize=True,max_length=self.config.model.length)
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    self._ratio_flat_num = 0
    self._ratio_flat_den = 0
    self._topk_vs_t    = {k: [] for k in self.TOPK_LIST}  # diffusion vs ratio
    self._topk_dg_vs_t = {k: [] for k in self.TOPK_LIST}  # diffusion vs guided
    self._t_records = []
    if self.ema:
      self.ema.store(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
      self.ema.copy_to(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    if self.guided_sampling and self.validation_mode:
      torch.cuda.reset_peak_memory_stats(self.device)
      if self.config.debug:
        torch.cuda.synchronize(self.device)
        alloc = torch.cuda.memory_allocated(self.device)/1e9
        resv  = torch.cuda.memory_reserved(self.device)/1e9
        peak  = torch.cuda.max_memory_allocated(self.device)/1e9
        print(f"[val_step {batch_idx}] alloc={alloc:.2f}GB reserved={resv:.2f}GB peak={peak:.2f}GB")
      return self._compute_guided_loss(batch)
    else:
      return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if self.guided_sampling and self.validation_mode:
        if self.trainer.is_global_zero and self.logger is not None and hasattr(self.logger, "experiment"):
            if not self._t_records:  # no valid overlap data this epoch
                for K in self.TOPK_LIST:
                  self.log(f"val/topk_overlap_ratio_source@{K}", float('nan'),
                          on_step=False, on_epoch=True, sync_dist=True)
                  self.log(f"val/topk_overlap_dvsg@{K}", float('nan'),
                          on_step=False, on_epoch=True, sync_dist=True)
            self._log_topk_table_wandb()
            pct = 100.0 * self._ratio_flat_num / self._ratio_flat_den
            # log epoch metrics
            self.log("val/ratio_flat_pct", pct, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("val/ratio_flat_count", float(self._ratio_flat_num), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("val/ratio_masked_total", float(self._ratio_flat_den), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(),
                                self.noise.parameters()))
        return
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      torch.cuda.empty_cache()          
      torch.cuda.reset_peak_memory_stats()
      for _ in range(1):#self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
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

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,}
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(self.config.eval.perplexity_batch_size,samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      # Splite the generated sample size into smaller sub batches  
      _samples = torch.split(samples[i * batch_size: (i + 1) * batch_size],eval_context_size,dim=-1)
      _attn_mask = torch.split(attn_mask[i * batch_size: (i + 1) * batch_size],eval_context_size,dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(_samples, _attn_mask):
        logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],sample_chunk[..., 1:],reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
        token_mask = (sample_chunk!= self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t- move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_denoise(
    self,
    xt: torch.tensor,
    time_conditioning: torch.tensor,
    move_chance_t: torch.tensor,
    move_chance_s: torch.tensor,
    cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:

    # Compute x_theta
    if cache is not None:
      log_x_theta = cache['log_x_theta']
    else:
      log_x_theta = self.forward(xt, time_conditioning)
      #if self.config.sampling.use_float64:
      #  log_x_theta = log_x_theta.to(torch.float64)
    x_theta = log_x_theta.exp()

    # Compute posterior
    if self.diffusion == 'absorbing_state':
      q_xs = x_theta * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      q_xs /= move_chance_t
    elif self.diffusion == 'uniform':
      q_xs = self._compute_posterior(
        x=x_theta,
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t)
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")
    # Sample from posterior
    xs = _sample_categorical(q_xs)
    if self.diffusion == 'absorbing_state':
      copy_flag = (xt != self.mask_index).to(torch.bool)
      q_xs[copy_flag] = 0.0
      q_xs[copy_flag, xt[copy_flag]] = 1.0
      xs = torch.where(copy_flag, xt, xs)

    return xs, q_xs, {'log_x_theta': log_x_theta}
    
  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    xt = self._sample_prior(batch_size_per_gpu,self.config.model.length).to(self.device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    pbar = tqdm(range(self.config.sampling.steps),desc='Sampling',leave=False)
    NFEs = 0
    cache = None
    q_xs = xt.new_zeros(1)

    for i in pbar:
      t, sigma_t, sigma_s, move_chance_t, move_chance_s = \
          self._compute_move_chances(timesteps[i], dt, xt.size(0))
      NFEs += 1 if cache is None else 0

      if self.sampler == 'ddpm_cache' and not self.config.sampling.guided_sampling:
        xs, q_xs, cache = self._ddpm_denoise(
          xt=xt,
          time_conditioning=sigma_t,
          move_chance_t=move_chance_t,
          move_chance_s=move_chance_s,
          cache=cache)
      elif self.sampler == 'ddpm_cache' and self.config.sampling.guided_sampling:
        xs, q_xs, cache = self._ratio_guidance_denoise(
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache)      
      else:
        xs = self._analytic_update(xt, t, dt)
      
      pbar.set_postfix(NFEs=NFEs,prob_check=(q_xs.sum() / xt.numel()).item(),nan_check=bool(q_xs.isnan().sum() > 0))
      if (not torch.allclose(xs, xt) or self.time_conditioning):
        cache = None
      xt = xs

    # just one last denoising step to get rid of remaining MASK tokens
    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(xt.shape[0], 1,device=self.device)
      if self.sampler == 'analytic':
        xs = self._denoiser_update(xt, t)
      else:
        unet_conditioning = self.noise(t)[0]
        xs = self.forward(xt, unet_conditioning).argmax(dim=-1)
    return xt

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
      self.ema.copy_to(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':     
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(model_output)
      unmasked_score = torch.scatter(unmasked_score,-1,x[..., None],torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
      model_output = (masked_score * masked_indices + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):
    t = self._sample_t(x0.shape[0])
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self._q_xt(x0, move_chance)
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(model_output=model_output, xt=xt, x0=x0, t=t)
      return diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(- torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(n_samples,self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x) or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((np.concatenate(intermediate_tokens, axis=1)[:, 1:] == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append( self.tokenizer.batch_decode(np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples, sequence_lengths)

  def _ratio_guidance_denoise(
      self,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
        ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
    seq_len = xt.shape[1]
    gamma = self.config.sampling.guidance_scale
    if cache is not None:
      log_x_theta = cache['log_x_theta']
      ratio_log = cache['ratio_log']
    else:
      # Diffusion model
      log_x_theta = self.forward(xt, time_conditioning)

      # Ratio model
      #xt_jumps = self._expand_with_single_token_replacements(xt=xt) # (B, L) -> (B · L · V, L) tensor.
      #batch_size = self.config.sampling.batch_size_ratio
      #ratio_log = self.ratio_model.get_log_probs(xt_jumps, batch_size ,time_conditioning.repeat(seq_len * self.vocab_size)) 
      number_sequences_in_a_chunk = self.config.sampling.batch_size_ratio
      ratio_log = self.get_ratio_log_stream(xt, time_conditioning, number_sequences_in_a_chunk)
      

    # Compute unguided posterior
    if self.diffusion == 'absorbing_state':
      diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
      diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      diffusion_log_probs.detach()
    elif self.diffusion == 'uniform':
      diffusion_log_probs = self._compute_posterior(x=log_x_theta.exp(),xt=xt,alpha_s=1 - move_chance_s,alpha_t=1 - move_chance_t).log()
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    # Apply guidance
    with torch.no_grad():
      if self.diffusion == 'absorbing_state':
        guided_log_probs = (gamma * ratio_log) + diffusion_log_probs
        copy_flag = (xt != self.mask_index)
        guided_log_probs[copy_flag] = self.neg_infinity
        guided_log_probs[copy_flag, xt[copy_flag]] = 0.0
      elif self.diffusion == 'uniform':
        guided_log_probs = (gamma * ratio_log) + diffusion_log_probs
      else:
        raise NotImplementedError(
          f"Diffusion type {self.diffusion} not implemented.")

    guided_probs = guided_log_probs.softmax(dim=-1)
    # Sample from guided posterior
    xs = _sample_categorical(guided_probs)
    if self.diffusion == 'absorbing_state':
      xs = torch.where(copy_flag.to(bool), xt, xs)
    return xs, guided_probs, {'log_x_theta': log_x_theta,
                              'ratio_log': ratio_log}
  
  def detokenize_batch(x0: torch.Tensor, tokenizer, *, skip_special_tokens: bool = True):
      """
      How to use:
      texts = self.detokenize_batch(x0, self.tokenizer)

      for i, txt in enumerate(texts[:3]):   # show first few examples
        print(f"[sample {i}] {txt}")
      """

      input_ids = x0.detach().cpu().tolist()
      return tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)
  
  def generative_perplexity_from_text(
    self,
    text_samples: typing.List[str],
    *,
    max_length: typing.Union[int, None] = None,
      ) -> float:
    """
    Return the generative perplexity of a batch of plain-text strings.

    It is a thin wrapper around `self.get_generative_perplexity`, so the
    heavy lifting (tokenisation, scoring, etc.) remains in one place.
    """
    return self.get_generative_perplexity(
        text_samples, retokenize=True, max_length=max_length
    )


  @torch.no_grad()
  def update_gen_ppl_metric_from_text(
    self,
    text_samples: typing.List[str],
    *,
    max_length: typing.Union[int, None] = None,
      ) -> float:
    """
    Compute NLLs + mask for a batch of text, update `self.gen_ppl_metric`,
    and return the corresponding perplexity.

    This is suitable for calling inside **training** or **validation** loops.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if max_length is None:
        max_length = self.config.model.length

    # 1. Load the (frozen) evaluator language model.
    scorer_name = self.gen_ppl_eval_model_name_or_path
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(scorer_name).eval()
    if "llama2" not in scorer_name:
        eval_model = eval_model.to(self.device)

    # 2. Re-tokenise with the scorer’s tokenizer.
    ids, attn_mask, ctx = self.eval_retokenize(text_samples, max_length=max_length)

    # 3. Walk through the sequence in context-length chunks.
    eos_id = self.eval_model_tokenizer.eos_token_id
    bsz = min(self.config.eval.perplexity_batch_size, ids.size(0))

    total_nll, total_tokens = 0.0, 0.0
    for i in range(0, ids.size(0), bsz):
        chunk_ids = ids[i : i + bsz]
        chunk_msk = attn_mask[i : i + bsz]

        for ids_split, msk_split in zip(
            torch.split(chunk_ids, ctx, dim=-1),
            torch.split(chunk_msk, ctx, dim=-1),
        ):
            logits = eval_model(
                ids_split.to(eval_model.device),
                attention_mask=msk_split.to(eval_model.device),
            ).logits.transpose(1, 2)  # (B, V, L)

            nlls = F.cross_entropy(
                logits[..., :-1], ids_split[..., 1:], reduction="none"
            )

            first_eos = (ids_split == eos_id).cumsum(-1) == 1
            token_mask = (ids_split != eos_id)
            mask = (first_eos | token_mask)[..., 1:]  # align with nlls

            # 4. Metric update expected by Lightning.
            self.gen_ppl_metric.update(nlls, mask)

            total_nll += (nlls * mask).sum().item()
            total_tokens += mask.sum().item()

    ppl = math.exp(total_nll / max(total_tokens, 1e-8))
    return ppl
  
  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),self.noise.parameters()))
      self.ema.copy_to(itertools.chain( self.backbone.parameters(), self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples, sequence_lengths) = self.sample_subs_guidance(
          n_samples=self.config.loader.eval_batch_size, stride_length=stride_length, num_strides=num_strides, dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
  
  def _expand_with_single_token_replacements(
    self, xt: torch.Tensor
  ) -> torch.Tensor:
    """
    Return a tensor where each row is `xt` with one token replaced
    by every vocabulary symbol, **but keep it on CPU**.
    """
    bsz, seq_len = xt.shape
    V = self.vocab_size

    # 1) build on CPU to avoid a huge GPU allocation
    xt_cpu = xt.to("cpu")                                     # (B, L)
    xt_expand = (xt_cpu.unsqueeze(1).repeat(1, seq_len * V, 1).view(-1, seq_len)                              # (B·L·V, L)
                 )

    # 2) overwrite the chosen positions
    jump_idx   = torch.arange(seq_len * V).repeat(bsz, 1).flatten()
    jump_dims  = jump_idx // V
    jump_state = jump_idx %  V
    xt_expand[torch.arange(jump_idx.size(0)), jump_dims] = jump_state

    return xt_expand          
  
  def _compute_move_chances(
    self,
    t_scalar: torch.Tensor,   # the current scalar timestep (shape: ())
    dt:       float,
    batch_sz: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """
    Return the tensors needed for a single sampling step.

    Outputs (same names you used before):
        t           – (B, 1)
        sigma_t     – (B,)
        sigma_s     – (B,)
        move_chance_t – (B, 1, 1)
        move_chance_s – (B, 1, 1)
    """
    # Map to training grid when using discrete-time diffusion.
    if self.T > 0:
        t_scalar = ((t_scalar * self.T).to(torch.int) / self.T) + 1 / self.T

    t = t_scalar.expand(batch_sz, 1).to(self.device)

    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    sigma_t = sigma_t.squeeze(-1)
    sigma_s = sigma_s.squeeze(-1)

    move_t = (1 - torch.exp(-sigma_t)).unsqueeze(-1).unsqueeze(-1)
    move_s = (1 - torch.exp(-sigma_s)).unsqueeze(-1).unsqueeze(-1)

    return t, sigma_t, sigma_s, move_t, move_s
  

  def batch_get_ratio_log_stream(self, xt, sigma, chunk_v: int = 1024):
    """
    Per-sample ratio scores over vocab **only at masked positions**.

    For each sample b and each position pos where xt[b,pos] == MASK,
    we evaluate the ratio model on n candidate substitutions in vocab
    chunks, gather the logit corresponding to the candidate token we
    actually injected, and write that scalar into ratio_log[b,pos,v].

    Unmasked positions receive 0.0 (log 1) so they are neutral when
    added (scaled) into guided_log_probs downstream.

    Args:
        xt:    (B, L) int64 tokens (any device).
        sigma: (B,) or (B,1) float timestep conditioning.
        chunk_v: vocab slice size processed per forward pass.

    Returns:
        ratio_log: (B, L, V) float32 tensor on ratio_model device.
                   Masked rows: log-softmaxed over vocab.
                   Unmasked rows: zeros.
    """
    device = xt.device
    B, L = xt.shape
    V = self.vocab_size

    # make sure sigma is 1D float tensor per batch item
    sigma = sigma.squeeze(-1) if sigma.ndim > 1 else sigma
    assert sigma.shape[0] == B, sigma.shape
    
    # allocate neutral log factors
    ratio_log = torch.zeros((B, L, V), device=device, dtype=torch.float32)

    xt_cpu = xt.detach().cpu()  # blocking copy; safe
    sigma_cpu = sigma.detach().cpu()

    for b in range(B):
        row = xt_cpu[b].clone()  # snapshot
        sig = sigma_cpu[b]          # scalar
        mask_pos = (row == self.mask_index).nonzero(as_tuple=False).flatten()
        if mask_pos.numel() == 0:
            continue  # nothing to do for this sample

        # loop masked positions
        for pos in tqdm(mask_pos, desc=f"Ratio {b}"):
            # fill vocab in chunks
            for v0 in range(0, V, chunk_v):
                v1 = min(v0 + chunk_v, V)
                n  = v1 - v0

                # build n mutated sequences (n, L)
                tmp = row.unsqueeze(0).repeat(n, 1)        # clone row n times
                tmp[:, pos] = torch.arange(v0, v1)         # inject candidates

                with torch.no_grad():
                  tmp_dev = tmp.to(device, non_blocking=True)
                  sig_dev = sig.repeat(n).to(device, non_blocking=True)
                  logits = self.ratio_model(tmp_dev, sig_dev)  # expect (n, L, V) or (n, V)


                # take logits for this position
                if logits.dim() == 3:
                    logits_pos = logits[:, pos, :]   # (n, V)
                else:
                    logits_pos = logits              # (n, V)
                cand_ids = torch.arange(v0, v1, device=logits_pos.device)  # (n,)
                cand_scores = logits_pos.gather(1, cand_ids.unsqueeze(1)).squeeze(1)
                ratio_log[b, pos, v0:v1] = cand_scores
            # normalize this (b,pos) row over full vocab
            ratio_log[b, pos, :] = torch.log_softmax(ratio_log[b, pos, :], dim=-1)

    return ratio_log
  

  def get_ratio_log_stream(self, xt, sigma, chunk_v: int = 1024):
    """
    Compute log-prob correction terms from the ratio model, but *only* for
    sequence positions that are currently MASK tokens in at least one item
    of the batch. Unmasked positions are left at 0.0 log-ratio (i.e., neutral).
    """
    B, L = xt.shape
    V = self.vocab_size

    # Handle DataParallel-wrapped ratio_model
    ratio_model = self.ratio_model.module if isinstance(self.ratio_model, torch.nn.DataParallel) else self.ratio_model
    device = next(ratio_model.parameters()).device
    param_dtype = next(ratio_model.parameters()).dtype

    # Allocate result (neutral log-factor = 0)
    ratio_log = torch.zeros((B, L, V), device=device, dtype=param_dtype)

    # Move inputs to CPU for expansion
    base_cpu  = xt.to("cpu", non_blocking=True)
    sigma_cpu = sigma.to("cpu", non_blocking=True)

    # Identify which columns contain at least one MASK
    mask_cols = (base_cpu == self.mask_index).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
    if len(mask_cols) == 0:
        # Nothing to guide; return neutral ratios.
        return ratio_log

    # Pre-build a [0..n-1] index once per chunk_v loop (built in loop below)
    for pos in tqdm(mask_cols, desc="ratio-mask-cols"):
        for v0 in range(0, V, chunk_v):
            v1 = min(v0 + chunk_v, V)
            n  = v1 - v0

            # (B, n, L) -> (B*n, L)
            cand = torch.arange(v0, v1, dtype=base_cpu.dtype, device=base_cpu.device)
            tmp = base_cpu.unsqueeze(1).expand(B, n, L).clone().reshape(-1, L)
            tmp[:, pos] = cand.repeat(B)

            tmp_dev = tmp.to(device, non_blocking=True)
            sig_dev = sigma_cpu.repeat_interleave(n).to(device, dtype=param_dtype, non_blocking=True)

            with  torch.inference_mode():
                logits = ratio_model(tmp_dev,sig_dev)

            # Expect (B*n, L, V) or (B*n, V_pos) fallback
            logits_pos = logits[:, pos, :] if logits.dim() == 3 else logits
            #if pos == mask_cols[0] and v0 == 0:
            #  print("DEBUG ratio_model out:", logits.shape)
            #  print("DEBUG logits_pos:", logits_pos.shape)
            cand_scores = logits_pos.reshape(B, n)   # works if logits_pos numel == B*n
            ratio_log[:, pos, v0:v1] = cand_scores
            del tmp_dev, sig_dev, logits_pos, logits, cand_scores

        # Normalize *this* column over vocab.
        ratio_log[:, pos, :] = torch.log_softmax(ratio_log[:, pos, :], dim=-1)
        torch.cuda.empty_cache()
    return ratio_log
  
  
  def _compute_guided_loss(self, batch: typing.Dict[str, torch.Tensor]) -> Loss:
    """
    Validation-time guided loss.

    We:
      4. Build move_chance from t -> generate a noised input xt.
      5. Run diffusion backbone -> log_x_theta (unnormalized log-scores).
      6. Query ratio model stream-wise -> ratio_log (logits over vocab).
      7. Combine: guided_log = log_x_theta + gamma * ratio_log.
      8. Log-softmax over vocab → guided_log_prob.

    Returns:
        Loss(dataclass): with token-averaged loss (.loss), per-token nll (.nlls),
        and mask (.token_mask), consistent with _compute_loss() so that Lightning
        logging doesn't break.
    """

    assert self.guided_sampling, "Called _compute_guided_loss but guided_sampling=False."
    device = self.device

    x0 = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask", torch.ones_like(x0, device=device))

    # Apply dataset-specific cropping / AR shift if configured.
    input_tokens, _, attention_mask = self._maybe_sub_sample(x0, attention_mask)

    B, L = input_tokens.shape
    gamma = self.guidance_scale

    # ---- 1. sample diffusion time (mirrors _forward_pass_diffusion) ----
    t = self._sample_t(B)  # (B,)
    if self.config.debug:
      t = torch.empty(B, device=device, dtype=self.dtype).uniform_(0.002, 0.003)
    if self.T > 0:
        t = (t * self.T).to(torch.int)
        t = t / self.T
        t = t + (1.0 / self.T)  # map to {1/T, ..., 1}

    # produce sigma & move_chance as in _forward_pass_diffusion
    if self.change_of_variables:
        time_conditioning = t[:, None]  # (B,1), in [0,1]
        f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
        move_chance = torch.exp(f_0 + t * (f_T - f_0))[:, None]  # (B,1)
    else:
        sigma, _dsigma = self.noise(t)  # (B,)
        time_conditioning = sigma[:, None]  # (B,1)
        move_chance = (1.0 - torch.exp(-sigma))[:, None]  # (B,1)

    # ---- 2. apply forward noising to create xt ----
    xt = self._q_xt(input_tokens, move_chance)  # (B,L)

    # ---- 3. diffusion model log-scores (logits) ----
    log_x_theta = self.forward(xt, time_conditioning)  # (B,L,V) log-scores (NOT softmaxed)

    # ---- 4. ratio model stream (returns logits over vocab) ----
    chunk_v = getattr(self.config.sampling, "batch_size_ratio", 1024)
    ratio_log = self.get_ratio_log_stream(xt, time_conditioning.squeeze(-1), chunk_v=chunk_v)
    # shape (B,L,V), raw logits from ratio model
    # 4b. top-K overlaps at masked positions ------------------------------------
    mask_positions = (xt == self.mask_index)
    s_diff  = log_x_theta[mask_positions]         # diffusion rows
    s_ratio = ratio_log[mask_positions]           # ratio rows

    flat_n, flat_d = self._ratio_flat_stats(ratio_log, xt, eps=1e-6)

    if flat_n != flat_d:  # at least *some* informative masked rows
        # Diffusion vs Ratio
        overlap_dr = self._topk_overlap_pct_batch(s_diff, s_ratio, self.TOPK_LIST, eps=1e-6)

        # Diffusion vs Guided (use pre-softmax; ranks identical to softmax)
        s_guided = (log_x_theta + gamma * ratio_log)[mask_positions]
        overlap_dg = self._topk_overlap_pct_batch(s_diff, s_guided, self.TOPK_LIST, eps=1e-6)

        # record t
        t_val = float(t.mean().item())
        self._t_records.append(t_val)

        # buffer + log (per-step)
        for K, pct in overlap_dr.items():
            self._topk_vs_t[K].append(pct)
            self.log(f"val/topk_overlap_ratio_source@{K}", pct,
                    on_step=False, on_epoch=True, sync_dist=True)

        for K, pct in overlap_dg.items():
            self._topk_dg_vs_t[K].append(pct)
            self.log(f"val/topk_overlap_dvsg@{K}", pct,
                    on_step=False, on_epoch=True, sync_dist=True)
    else:
        # fully flat => skip; leave buffers unchanged, but still count flats below
        pass
      
    self._ratio_flat_num += flat_n
    self._ratio_flat_den += flat_d

    # ---- 5. combine diffusion + ratio guidance ----
    guided_log = log_x_theta + gamma * ratio_log  # still unnormalized

    # ---- 6. normalize over vocab ----
    guided_log_prob = torch.log_softmax(guided_log, dim=-1)  # (B,L,V)

    # ---- 7. per-token NLL against clean tokens ----
    # gather log prob at ground-truth token
    tgt = input_tokens.unsqueeze(-1)  # (B,L,1)
    logp_tgt = torch.gather(guided_log_prob, dim=-1, index=tgt).squeeze(-1)  # (B,L)
    nlls = -logp_tgt

    # ---- 8. mask + reduce ----
    nlls = nlls * attention_mask
    denom = attention_mask.sum()
    token_nll = nlls.sum() / torch.clamp_min(denom, 1)

    # ---- 9. update validation metrics (so Lightning logs) ----
    # Use the same prefixing scheme as _compute_loss('val')
    self.valid_metrics.update(nlls, attention_mask)
    self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, sync_dist=True)

    return Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)

  def _topk_overlap_pct_batch(self, s1, s2, k_list, eps=1e-8):
    out = {}
    V = s1.size(-1)
    # drop constant rows in s2
    keep = (s2.max(-1).values - s2.min(-1).values) > eps
    if not keep.any():
        for K in k_list:
            out[K] = float('nan')
        return out
    s1 = s1[keep]; s2 = s2[keep]
    for K in k_list:
        k = min(K, V)
        idx1 = torch.topk(s1, k, dim=-1).indices
        idx2 = torch.topk(s2, k, dim=-1).indices
        isin = torch.isin(idx1, idx2) if hasattr(torch, "isin") else (idx1.unsqueeze(-1) == idx2.unsqueeze(-2)).any(-1)
        pct = (isin.sum(-1).float() / k * 100.0).mean().item()
        out[K] = pct
    return out
        
  def _ratio_flat_stats(self, ratio_log: torch.Tensor, xt: torch.Tensor, eps: float = 1e-8):
    """Return (#flat_rows, #masked_rows) for this batch."""
    mask = (xt == self.mask_index)
    if not mask.any():
        return 0, 0
    rows = ratio_log[mask]  # (N,V)
    if rows.numel() == 0:
        return 0, 0
    span = rows.max(dim=-1).values - rows.min(dim=-1).values
    flat = (span < eps)
    return int(flat.sum().item()), int(flat.numel())
  
def _log_topk_table_wandb(self):
    """Push top-k overlap vs t tables to W&B (if available & rank0)."""
    if not getattr(self.trainer, "is_global_zero", False):
        return
    exp = getattr(self.logger, "experiment", None)
    if exp is None:
        return
    try:
        import wandb
    except ImportError:
        return
    if not self._t_records:
        return  # nothing to plot

    def _mk_rows(buf_dict):
        rows = []
        for i, t_val in enumerate(self._t_records):
            row = [t_val]
            for k in self.TOPK_LIST:
                lst = buf_dict[k]
                row.append(lst[i] if i < len(lst) else float("nan"))
            rows.append(row)
        return rows

    # diffusion vs ratio
    cols_dr = ["t"] + [f"topk@{k}" for k in self.TOPK_LIST]
    tbl_dr = wandb.Table(columns=cols_dr, data=_mk_rows(self._topk_vs_t))
    exp.log({"val/topk_vs_t": tbl_dr})

    # diffusion vs guided
    cols_dg = ["t"] + [f"topk_dvsg@{k}" for k in self.TOPK_LIST]
    tbl_dg = wandb.Table(columns=cols_dg, data=_mk_rows(self._topk_dg_vs_t))
    exp.log({"val/topk_dvsg_vs_t": tbl_dg})

    # optional line plots
    for k in self.TOPK_LIST:
        exp.log({
            f"val/topk_vs_t_{k}": wandb.plot.line(tbl_dr, "t", f"topk@{k}",
                                                  title=f"Top-{k} overlap (diffusion vs ratio)"),
            f"val/topk_dvsg_vs_t_{k}": wandb.plot.line(tbl_dg, "t", f"topk_dvsg@{k}",
                                                       title=f"Top-{k} overlap (diffusion vs guided)"),
        })