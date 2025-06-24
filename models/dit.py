import math
import typing
# https://github.com/facebookresearch/DiT/blob/main/models.py
# https://arxiv.org/pdf/2212.09748
try:
    import flash_attn
    from flash_attn.layers.rotary import apply_rotary_emb_qkv_
    has_flash = True
except ModuleNotFoundError:
    has_flash = False

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def apply_rotary_pos_emb_for_macos(qkv, cos, sin):
  """
  qkv : (B, S, 3, H, D)
  cos/sin : (1, S, 3, 1, D)
  returns qkv with rotary applied to Q and K.
  """
  # slice out the (S, D/2) tensors exactly like the FlashAttention path
  cos_slice = cos[0, :, 0, 0, : cos.shape[-1] // 2]  # (S, D/2)
  sin_slice = sin[0, :, 0, 0, : sin.shape[-1] // 2]

  if has_flash:  # fast CUDA kernel
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv,
                                                          cos_slice,
                                                          sin_slice)

  # ------------------------------------------------------------------
  # CPU / MPS fallback: broadcast cos/sin so they match (B, S, H, D/2)
  cos_b = cos_slice.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
  sin_b = sin_slice.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)

  q, k, v = qkv.unbind(dim=2)  # each (B, S, H, D)

  def _rot(vec):
    v1, v2 = vec[..., : vec.shape[-1] // 2], vec[..., vec.shape[-1] // 2:]
    # (B, S, H, D/2) each
    return torch.cat([v1 * cos_b - v2 * sin_b,
                      v1 * sin_b + v2 * cos_b], dim=-1)

  q_rot, k_rot = _rot(q), _rot(k)
  return torch.stack([q_rot, k_rot, v], dim=2)  # (B, S, 3, H, D)


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1, use_adaLN: bool = True):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.use_adaLN = use_adaLN

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.use_adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    else:
      self.register_buffer("adaLN_modulation", None)


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    if self.use_adaLN and c is not None:
      (shift_msa, scale_msa, gate_msa,
       shift_mlp, scale_mlp, gate_mlp) = (
          self.adaLN_modulation(c)[:, None].chunk(6, dim=2))
    else:
      zeros = torch.zeros(batch_size, 1, self.dim,
                          device=x.device, dtype=x.dtype)
      ones = torch.ones_like(zeros)
      shift_msa = scale_msa = shift_mlp = scale_mlp = zeros
      gate_msa = gate_mlp = ones

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=False):
      cos, sin = rotary_cos_sin
      if has_flash:
        qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_for_macos(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    if has_flash:
      qkv = rearrange(qkv, 'b s ... -> (b s) ...')
      if seqlens is None:
        cu_seqlens = torch.arange(
          0, (batch_size + 1) * seq_len, step=seq_len,
          dtype=torch.int32, device=qkv.device)
      else:
        cu_seqlens = seqlens.cumsum(-1)

      # (b s, 3, h, d) → FlashAttention
      qkv_flat = rearrange(qkv, 'b s three h d -> (b s) three h d')
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv_flat, cu_seqlens, seq_len, 0.0, causal=False)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      # split then call PyTorch fused SDP
      qkv = rearrange(qkv, 'b s three h d -> b s three h d')  # no flattening
      q, k, v = qkv.unbind(dim=2)  # (b, s, h, d)
      q = rearrange(q, 'b s h d -> b h s d')
      k = rearrange(k, 'b s h d -> b h s d')
      v = rearrange(v, 'b s h d -> b h s d')
      x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
      x = rearrange(x, 'b h s d -> b s (h d)')



    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    if c is None:
      shift = scale = torch.zeros(x.size(0), 1, x.size(-1),
                                  device=x.device, dtype=x.dtype)
    else:
      shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  # https://github.com/facebookresearch/DiT/blob/main/models.py
  # https://arxiv.org/pdf/2212.09748
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))  # use_adaLN defaults to True
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices, sigma):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    return x
class DITClassifier(nn.Module):
  def __init__(self, config, vocab_size, time_conditioning=False):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.time_dependent = time_conditioning

    self.vocab_embed = EmbeddingLayer(
      config.classifier_model.hidden_size, vocab_size)

    if not self.time_dependent:
      self.sigma_map = None
    else:
      self.sigma_map = TimestepEmbedder(config.classifier_model.cond_dim)

    self.rotary_emb = Rotary(
      config.classifier_model.hidden_size // config.classifier_model.n_heads)

    blocks = []
    for _ in range(config.classifier_model.n_blocks):
      blocks.append(
        DDiTBlock(config.classifier_model.hidden_size,
                  config.classifier_model.n_heads,
                  config.classifier_model.cond_dim,
                  dropout=config.classifier_model.dropout,
                  use_adaLN=self.time_dependent))
    self.blocks = nn.ModuleList(blocks)

    self.scale_by_sigma = config.classifier_model.scale_by_sigma

    self.pooling = getattr(config.classifier_model, 'pooling', 'mean')
    self.output_layer = nn.Linear(
      config.classifier_model.hidden_size,
      1)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices_or_one_hots, sigma=None, x_emb=None, attention_mask=None):
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float),
                     self.vocab_embed.embedding.T)

      if not self.time_dependent:
        c = None
      else:
        c = F.silu(self.sigma_map(sigma))

      rotary_cos_sin = self.rotary_emb(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks)):
          x = self.blocks[i](x, rotary_cos_sin, c,
                             seqlens=None)
    else:
      x = x_emb

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      if self.pooling == 'mean':
        x = x.mean(dim=1)
      elif self.pooling == 'max':
        x = x.max(dim=1)
      elif self.pooling == 'cls':
        x = x[..., 0]
      elif self.pooling == 'last':
        x = x[..., -1]
      elif self.pooling == 'no_pooling':  # for ar_fudge
        pass
      elif self.pooling == 'attention_mean':  # for ar_pplm
        masked_x = x * attention_mask.unsqueeze(2)
        x = torch.sum(masked_x, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-15)
      else:
        raise NotImplementedError(
          f"`{self.pooling}` method not implemented.")
      x = self.output_layer(x)
    return x

  def load_pretrained_encoder(self, encoder: nn.Module):
    self.vocab_embed = encoder.vocab_embed
    self.sigma_map = encoder.sigma_map
    self.rotary_emb = encoder.rotary_emb
    self.blocks = encoder.blocks


class DITRatio(nn.Module):
  def __init__(self, config, vocab_size, time_conditioning=False):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.time_dependent = time_conditioning

    self.vocab_embed = EmbeddingLayer(
      config.ratio_model.hidden_size, vocab_size)

    if not self.time_dependent:
      self.sigma_map = None
    else:
      self.sigma_map = TimestepEmbedder(config.ratio_model.cond_dim)

    self.rotary_emb = Rotary(
      config.classifier_model.hidden_size // config.ratio_model.n_heads)

    blocks = []
    for _ in range(config.ratio_model.n_blocks):
      blocks.append(
        DDiTBlock(config.ratio_model.hidden_size,
                  config.ratio_model.n_heads,
                  config.ratio_model.cond_dim,
                  dropout=config.ratio_model.dropout,
                  use_adaLN=self.time_dependent))
    self.blocks = nn.ModuleList(blocks)

    self.scale_by_sigma = config.ratio_model.scale_by_sigma

    self.pooling = getattr(config.ratio_model, 'pooling', 'mean')
    self.output_layer = nn.Linear(
      config.ratio_model.hidden_size,
      config.ratio_model.num_classes)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices_or_one_hots, sigma=None, x_emb=None, attention_mask=None):
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float),
                     self.vocab_embed.embedding.T)

      if not self.time_dependent:
        c = None
      else:
        c = F.silu(self.sigma_map(sigma))

      rotary_cos_sin = self.rotary_emb(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks)):
          x = self.blocks[i](x, rotary_cos_sin, c,
                             seqlens=None)
    else:
      x = x_emb

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      if self.pooling == 'mean':
        x = x.mean(dim=1)
      elif self.pooling == 'max':
        x = x.max(dim=1)
      elif self.pooling == 'cls':
        x = x[..., 0]
      elif self.pooling == 'last':
        x = x[..., -1]
      elif self.pooling == 'no_pooling':  # for ar_fudge
        pass
      elif self.pooling == 'attention_mean':  # for ar_pplm
        masked_x = x * attention_mask.unsqueeze(2)
        x = torch.sum(masked_x, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-15)
      else:
        raise NotImplementedError(
          f"`{self.pooling}` method not implemented.")
      x = self.output_layer(x)
    return x

  def load_pretrained_encoder(self, encoder: nn.Module):
    self.vocab_embed = encoder.vocab_embed
    self.sigma_map = encoder.sigma_map
    self.rotary_emb = encoder.rotary_emb
    self.blocks = encoder.blocks