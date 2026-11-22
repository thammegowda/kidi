#!/usr/bin/env python3
"""Readable PyTorch reference for the Inkling-2 text model, including its loop.

This is the specification and layer-dump oracle for Kidi's `inkling2` port. It
recomputes the whole sequence on every call (no KV/conv caches), which keeps the
looped core easy to follow: each visit of a core layer attends only over keys of
the same pass, which is exactly what per-pass caches hold.

Sources: transformers `models/inkling2/modeling_inkling2.py` (non-loop layers)
and vLLM `models/inkling2/nvidia/loop.py` (loop schedule and message blending).

    python tools/reference/inkling2.py --model <snapshot> --prompt "Hello" --tokens 16
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
STREAMS = ("attn", "k", "v", "mlp")


def dequantize_nvfp4(packed, scale, amax, denominator):
    """[E, N, K/2] U8 (low nibble first) with [E, N, K/16] E4M3 block scales -> BF16 [E, N, K]."""
    lut = torch.tensor(E2M1, device=packed.device)
    values = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1)
    values = values.reshape(*packed.shape[:2], -1, 16) * scale.float()[..., None]
    global_scale = (amax.float() / denominator.float()).reshape(-1, 1, 1)
    return (values.reshape(*packed.shape[:2], -1) * global_scale).to(torch.bfloat16)


def dequantize_mxfp8(weight, scale):
    """[E, N, K] E4M3 with [E, N, K/32] UE8M0 block exponents -> BF16 [E, N, K]."""
    exponents = torch.exp2(scale.float() - 127.0)
    values = weight.float().reshape(*weight.shape[:2], -1, 32) * exponents[..., None]
    return values.reshape(weight.shape).to(torch.bfloat16)


def rms(x, eps=1e-6):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def rms_norm(x, weight, eps=1e-6):
    return (rms(x, eps) * weight.float()).to(torch.bfloat16)


def l2_clip(x, threshold, scale=None):
    x = x.float() if scale is None else x.float() * scale.float()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return (x * (threshold / (norm + 1e-6)).clamp(max=1.0)).to(torch.bfloat16)


def softcap(x, cap):
    return torch.tanh(x / cap) * cap


def linear(x, weight):
    return x.to(torch.bfloat16) @ weight.to(torch.bfloat16).T


def mix(sources, queries):
    """Block attention residual: softmax over RMS-normalized sources with pseudo-queries.

    sources: list of [T, D]; queries: [Q, D]. Returns FP32 [Q, T, D] (the mixed values are normalized too).
    """
    stacked = rms(torch.stack([s.float() for s in sources]))  # [N, T, D]
    logits = torch.einsum("qd,ntd->qtn", queries.float(), stacked) * stacked.shape[-1] ** -0.5
    return torch.einsum("qtn,ntd->qtd", logits.softmax(-1), stacked)


def short_conv(x, weight):
    """Causal depthwise conv, `weight` [C, 1, W] with the last tap on the current token, plus residual."""
    width = weight.shape[-1]
    x32 = x.float()
    y = F.conv1d(x32.T[None], weight.float(), padding=width - 1, groups=x.shape[-1])[0, :, : x.shape[0]].T
    return ((y + x32) / math.sqrt(width + 1)).to(torch.bfloat16)


def dynamic_conv(x, static_w, taps, heads):
    """Per-token taps: `static_w` [W, C] (tap 0 = current token) + `taps` [T, heads * W] shared per head."""
    width, channels = static_w.shape
    tokens = x.shape[0]
    x32 = x.float()
    weights = static_w.float().T.reshape(heads, channels // heads, width)
    weights = weights[None] + taps.float().reshape(tokens, heads, 1, width)  # [T, H, C/H, W]
    padded = F.pad(x32.T, (width - 1, 0))  # [C, T + W - 1]
    windows = padded.unfold(-1, width, 1).flip(-1).permute(1, 0, 2)  # [T, C, W], lag 0 first
    y = (windows.reshape(tokens, heads, -1, width) * weights).sum(-1).reshape(tokens, channels)
    return ((y + x32) / math.sqrt(2 * width + 1)).to(torch.bfloat16)


class Inkling2:
    def __init__(self, model_dir, device="cuda"):
        model_dir = Path(model_dir)
        root = json.loads((model_dir / "config.json").read_text())
        c = self.config = root.get("text_config", root)
        self.eos = root.get("eos_token_id", c.get("eos_token_id"))
        self.device = device
        self.hidden = c["hidden_size"]
        self.layers = c["num_hidden_layers"]
        self.heads = c["num_attention_heads"]
        self.kv_heads = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.local = set(c["local_layer_ids"])
        self.window = c["sliding_window_size"]
        self.rel_extent = c["rel_extent"]
        self.block = c["n_layers_per_attn_res_block"]
        self.loop = c.get("loop_config")
        self.vocab = c.get("unpadded_vocab_size") or c["vocab_size"]
        self.file = safe_open(str(model_dir / "model-00001-of-00001.safetensors"), "pt", device=device)
        self.prefix = "model.llm."
        self.cache = {}

    def w(self, name):
        if name not in self.cache:
            self.cache[name] = self.file.get_tensor(self.prefix + name)
        return self.cache[name]

    def experts(self, layer):
        """Dequantized BF16 expert weights `w13` [E, 2I, D] (gate/up rows interleaved) and `w2` [E, D, I]."""
        p = f"{self.prefix}layers.{layer}.mlp.experts."
        out = []
        for name in ("w13", "w2"):
            weight = self.file.get_tensor(p + name + "_weight")
            scale = self.file.get_tensor(p + name + "_weight_scale")
            if weight.dtype == torch.uint8:
                amax = self.file.get_tensor(p + name + "_weight_amax")
                denominator = self.file.get_tensor(p + name + "_weight_outer_scale_denominator")
                out.append(dequantize_nvfp4(weight, scale, amax, denominator))
            else:
                out.append(dequantize_mxfp8(weight, scale))
        return out

    def attention(self, i, xq, xk, xv, positions):
        p = f"layers.{i}.attn."
        c = self.config
        local = i in self.local
        tokens = xq.shape[0]
        q = linear(xq, self.w(p + "wq_du.weight")).view(tokens, self.heads, self.head_dim)
        r = linear(xq, self.w(p + "wr_du.weight")).view(tokens, self.heads, -1)
        gate = linear(xq, self.w(p + "gate_attn.weight"))
        sink = linear(xq, self.w(p + "attn_sink_proj.weight")).float()  # [T, H]
        k = linear(xk, self.w(p + "wk_dv.weight"))
        v = linear(xv, self.w(p + "wv_dv.weight"))
        if local:
            k = short_conv(k, self.w(p + "k_sconv.weight"))
            v = short_conv(v, self.w(p + "v_sconv.weight"))
        else:
            k_taps = linear(xk, self.w(p + "k_sconv.weight_proj"))
            v_taps = linear(xv, self.w(p + "v_sconv.weight_proj"))
            k = dynamic_conv(k, self.w(p + "k_sconv.static_w"), k_taps, self.kv_heads)
            v = dynamic_conv(v, self.w(p + "v_sconv.static_w"), v_taps, self.kv_heads)
        q = l2_clip(q, c["qk_l2_clip_threshold"], self.w(p + "q_l2_clip.scale"))
        k = l2_clip(k.view(tokens, self.kv_heads, self.head_dim), c["qk_l2_clip_threshold"])
        v = rms(v.view(tokens, self.kv_heads, self.head_dim)).to(torch.bfloat16)
        r = l2_clip(r, c["rel_l2_clip_threshold"], self.w(p + "rel_l2_clip.scale"))
        if not local:
            tau = 1 + c["log_scaling_alpha"] * torch.log(((positions + 1).float() / c["log_scaling_n_floor"]).clamp(min=1))
            q = (q.float() * tau[:, None, None]).to(torch.bfloat16)
            r = (r.float() * tau[:, None, None]).to(torch.bfloat16)

        extent = self.window if local else self.rel_extent
        distance = positions[:, None] - positions[None, :]  # [Tq, Tk]
        rel = (r.float() @ self.w(p + "rel_logits_proj.proj").float()).permute(1, 0, 2)  # [H, T, extent]
        bias = rel.gather(-1, distance.clamp(0, extent - 1).expand(self.heads, -1, -1))
        bias = bias.masked_fill((distance < 0) | (distance >= extent), 0.0)
        allowed = distance >= 0
        if local:
            allowed &= distance < self.window

        group = self.heads // self.kv_heads
        keys = k.float().repeat_interleave(group, 1).permute(1, 2, 0)  # [H, D, T]
        values = v.float().repeat_interleave(group, 1).transpose(0, 1)  # [H, T, D]
        scores = q.float().transpose(0, 1) @ keys / math.sqrt(self.head_dim) + bias
        scores = scores.masked_fill(~allowed, float("-inf"))
        scores = torch.cat([scores, sink.T[:, :, None]], -1)  # the sink takes probability mass with a zero value
        probs = scores.softmax(-1)[..., :-1]
        out = (probs @ values).transpose(0, 1).reshape(tokens, -1)
        out = (out * torch.sigmoid(gate.float())).to(torch.bfloat16)
        return linear(out, self.w(p + "wo_ud.weight"))

    def moe(self, i, x):
        p = f"layers.{i}.mlp."
        c = self.config
        scores = torch.sigmoid(x.float() @ self.w(p + "gate.weight").float().T)
        top = torch.topk(scores + self.w(p + "gate.bias").float(), c["num_experts_per_tok"], dim=-1).indices
        weights = scores.gather(1, top)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * c["route_scale"]
        w13, w2 = self.experts(i)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for expert in top.unique().tolist():
            token, slot = torch.where(top == expert)
            gate_up = linear(x[token], w13[expert]).float()
            g, u = gate_up[:, 0::2], gate_up[:, 1::2]
            act = softcap(g, c["mlp_gate_softcap"]) * torch.sigmoid(g) * softcap(u, c["mlp_up_softcap"])
            act = (act * weights[token, slot, None]).to(torch.bfloat16)
            out.index_add_(0, token, linear(act, w2[expert]).float())
        return out.to(torch.bfloat16)

    def blend(self, i, stream, dest, message, norm_weight):
        """Loop controller: gated low-rank merge of a message into an FP32 sublayer input, then RMSNorm."""
        p = f"layers.{i}.controllers.{stream}."
        pair = torch.cat([rms(dest), rms(message)], -1).to(torch.bfloat16)
        hidden = linear(pair, self.w(p + "down_weight"))
        gain = hidden.float() @ self.w(p + "gain_weight").float().T
        forget = hidden.float() @ self.w(p + "forget_weight").float().T
        return rms_norm((1 - torch.tanh(forget)) * dest + torch.tanh(gain) * message, norm_weight)

    def message_queries(self, i):
        """Pseudo-queries of the (attn, k, v, mlp) messages of core layer `i`."""
        j = i - self.loop["core_start"]
        queries = self.w("middle_cycle_cross_pass_queries")
        kv = self.w("middle_cycle_cross_pass_kv_queries")
        return torch.stack([queries[2 * j], kv[2 * j], kv[2 * j + 1], queries[2 * j + 1]])

    def visit(self, i, state, positions, bank=None, trace=None):
        """Run layer `i` on `state` = (completed blocks, partial block); `bank` is set in loop passes after the first."""
        blocks, partial = state
        p = f"layers.{i}."
        local = i in self.local
        norms = [self.w(p + "attn_norm.weight")] * 3 if local else [self.w(p + f"{n}_input_norm.weight") for n in "qkv"]
        messages = None if bank is None else mix(bank, self.message_queries(i))
        if i == 0:
            inputs = [rms_norm(partial, norms[0])] * 3
        else:
            # Full layers mix separate q/k/v inputs; local layers share one mix.
            dests = mix(blocks + [partial], self.w(f"attn_res.attn_queries.{i - 1}").view(-1, self.hidden))
            dests = dests.expand(3, -1, -1)
            if messages is None:
                inputs = [rms_norm(dests[s], norms[s]) for s in range(3)]
            else:
                inputs = [self.blend(i, STREAMS[s], dests[s], messages[s], norms[s]) for s in range(3)]
        attn_out = short_conv(self.attention(i, *inputs, positions), self.w(p + "attn_sconv.weight"))

        if i % self.block == 0:
            blocks = blocks + [partial]
            residual = attn_out.float()
        else:
            residual = partial.float() + attn_out.float()
        mlp_mix = mix(blocks + [residual], self.w(f"attn_res.mlp_queries.{i}").view(1, -1))[0]
        if messages is None:
            mlp_in = rms_norm(mlp_mix, self.w(p + "mlp_norm.weight"))
        else:
            mlp_in = self.blend(i, "mlp", mlp_mix, messages[3], self.w(p + "mlp_norm.weight"))
        mlp_out = short_conv(self.moe(i, mlp_in), self.w(p + "mlp_sconv.weight"))
        partial = (residual + mlp_out.float()).to(torch.bfloat16)
        if trace is not None:
            trace.append({"layer": i, "attn_out": attn_out, "mlp_in": mlp_in, "partial": partial})
        return blocks, partial

    @torch.no_grad()
    def forward(self, tokens, trace=None):
        """FP32 softcapped logits [T, vocab] for a token list."""
        ids = torch.tensor(tokens, device=self.device)
        positions = torch.arange(len(tokens), device=self.device)
        x = rms_norm(self.w("embed.weight")[ids], self.w("embed_norm.weight"))
        state = ([], x)
        core = range(self.loop["core_start"], self.loop["core_end"]) if self.loop else range(self.layers, self.layers)
        for i in range(core.start):
            state = self.visit(i, state, positions, trace=trace)
        if core:
            assert core.start % self.block == 0, "the prelude must end on an AttnRes block boundary"
            prelude = state
            for loop_pass in range(self.loop["num_passes"]):
                bank = None
                if loop_pass:
                    # The previous pass's blocks after the prelude's, then its partial; every pass restarts there.
                    blocks, partial = state
                    bank = blocks[len(prelude[0]) + 1 :] + [partial]
                    state = prelude
                for i in core:
                    state = self.visit(i, state, positions, bank, trace)
        for i in range(core.stop, self.layers):
            state = self.visit(i, state, positions, trace=trace)
        blocks, partial = state
        out = mix(blocks + [partial], self.w("attn_res.output_query").view(1, -1))[0]
        logits = linear(rms_norm(out, self.w("norm.weight")), self.w("unembed.weight")).float()
        return softcap(logits, self.config["final_logit_softcapping"])[:, : self.vocab]

    def generate(self, tokens, count):
        tokens = list(tokens)
        for _ in range(count):
            token = int(self.forward(tokens)[-1].argmax())
            tokens.append(token)
            if token == self.eos:
                break
        return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Inkling-2 Hugging Face snapshot directory")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    model = Inkling2(args.model, args.device)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    output = model.generate(prompt, args.tokens)[len(prompt) :]
    print(json.dumps({"prompt": prompt, "output": output, "text": tokenizer.decode(output)}))


if __name__ == "__main__":
    main()
