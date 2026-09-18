"""Explicit, ONNX-exportable LFM2 inference with immutable hybrid cache tensors.

One graph serves both prefill and field scoring. Attention cache slot zero is
always masked: this avoids zero-sized tensors in WebGPU on the first invocation.
The last valid token is projected directly to the trained answer logits.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PortableDecision(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.choice = model.choice
        cfg = self.backbone.config
        assert cfg.rope_parameters['rope_type'] == 'default'
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.state_names = []
        self.state_shapes = []
        self.attention_state = None
        for i, layer in enumerate(self.backbone.layers):
            if layer.is_attention_layer:
                if self.attention_state is None:
                    self.attention_state = len(self.state_names)
                self.state_names.extend([f'key_{i}', f'value_{i}'])
                self.state_shapes.extend([[1, self.kv_heads, 1, self.head_dim]] * 2)
            else:
                self.state_names.append(f'conv_{i}')
                self.state_shapes.append([1, cfg.hidden_size, layer.conv.conv_kernel_size])

    def empty_cache(self):
        param = next(self.backbone.parameters())
        return tuple(torch.zeros(shape, dtype=param.dtype, device=param.device) for shape in self.state_shapes)

    @staticmethod
    def rotate(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)

    def forward(self, input_ids, attention_mask, sizes, *states):
        h = self.backbone.embed_tokens(input_ids)
        batch, length, _ = h.shape
        past = states[self.attention_state].shape[-2] - 1
        positions = torch.arange(length, device=h.device) + past
        frequency = positions.float()[:, None] * self.backbone.rotary_emb.inv_freq.float()[None, :]
        angles = torch.cat((frequency, frequency), dim=-1)
        cos, sin = angles.cos().to(h.dtype)[None, None], angles.sin().to(h.dtype)[None, None]
        keys = torch.arange(past + length + 1, device=h.device)
        causal = (keys[None, :] <= positions[:, None] + 1) & (keys[None, :] > 0)
        valid = torch.cat((torch.ones((batch, past + 1), dtype=attention_mask.dtype, device=h.device), attention_mask), dim=1)
        allowed = causal[None, None, :, :] & (valid[:, None, None, :] > 0)
        mask = torch.where(allowed, torch.zeros((), dtype=h.dtype, device=h.device),
                           torch.full((), torch.finfo(h.dtype).min, dtype=h.dtype, device=h.device))
        index, updated = 0, []
        for layer in self.backbone.layers:
            x = layer.operator_norm(h)
            if layer.is_attention_layer:
                attn = layer.self_attn
                q = attn.q_layernorm(attn.q_proj(x).reshape(batch, length, self.heads, self.head_dim)).transpose(1, 2)
                k = attn.k_layernorm(attn.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)).transpose(1, 2)
                v = attn.v_proj(x).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
                q, k = q * cos + self.rotate(q) * sin, k * cos + self.rotate(k) * sin
                k = torch.cat((states[index].expand(batch, -1, -1, -1), k), dim=-2)
                v = torch.cat((states[index + 1].expand(batch, -1, -1, -1), v), dim=-2)
                updated.extend((k, v)); index += 2
                groups = self.heads // self.kv_heads
                k = k[:, :, None].expand(batch, self.kv_heads, groups, -1, self.head_dim).reshape(batch, self.heads, -1, self.head_dim)
                v = v[:, :, None].expand(batch, self.kv_heads, groups, -1, self.head_dim).reshape(batch, self.heads, -1, self.head_dim)
                weights = (q @ k.transpose(-1, -2)) * attn.scaling + mask
                weights = weights.float().softmax(-1).to(q.dtype)
                x = attn.out_proj((weights @ v).transpose(1, 2).reshape(batch, length, -1))
            else:
                conv = layer.conv
                x = x * attention_mask[:, :, None].to(x.dtype)
                b, c, z = conv.in_proj(x).transpose(1, 2).chunk(3, dim=1)
                z = torch.cat((states[index].expand(batch, -1, -1), b * z), dim=-1)
                updated.append(z[:, :, -conv.conv_kernel_size:]); index += 1
                y = F.conv1d(z, conv.conv.weight, conv.conv.bias,
                             padding=conv.conv_kernel_size - 1, groups=z.shape[1])
                y = y[:, :, :z.shape[-1]][:, :, -length:]
                x = conv.out_proj((c * y).transpose(1, 2))
            h = h + x
            h = h + layer.feed_forward(layer.ffn_norm(h))
        h = self.backbone.embedding_norm(h)
        lengths = attention_mask.to(torch.int64).sum(-1)
        # Gather, instead of advanced indexing, keeps the exported graph portable.
        pooled = torch.gather(h, 1, (lengths - 1)[:, None, None].expand(-1, 1, h.shape[-1])).squeeze(1)
        logits = self.choice(pooled.float())
        logits = logits.masked_fill(torch.arange(self.choice.out_features, device=h.device)[None] >= sizes[:, None], -1e4)
        return (logits, *updated)
