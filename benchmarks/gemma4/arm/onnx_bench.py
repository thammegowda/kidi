"""Export and benchmark Gemma 4 E2B on ONNX Runtime CPU (MLAS + Arm KleidiAI)."""

import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np


def peak_rss_kib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss // 1024 if sys.platform == "darwin" else rss


def emit(record):
    record["peak_rss_kib"] = peak_rss_kib()
    print(json.dumps(record), flush=True)


def build_onnx_model(shared_dir: Path, onnx_path: Path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from safetensors import safe_open

    config = json.loads((shared_dir / "config.json").read_text())
    text_cfg = config["text_config"]
    quant_cfg = config["quantization_config"]
    bits = int(quant_cfg["num_bits"])
    group_size = int(quant_cfg.get("group_size", 0))

    hidden_size = int(text_cfg["hidden_size"])
    num_layers = int(text_cfg["num_hidden_layers"])
    num_kv_shared = int(text_cfg["num_kv_shared_layers"])
    num_self_layers = num_layers - num_kv_shared
    num_heads = int(text_cfg["num_attention_heads"])
    head_dim_sliding = int(text_cfg["head_dim"])
    head_dim_full = int(text_cfg["global_head_dim"])
    ple_dim = int(text_cfg["hidden_size_per_layer_input"])
    eps = float(text_cfg["rms_norm_eps"])
    softcap = float(text_cfg["final_logit_softcapping"])
    layer_types = list(text_cfg["layer_types"])

    data_path = onnx_path.with_suffix(".onnx.data")
    if onnx_path.exists():
        onnx_path.unlink()
    if data_path.exists():
        data_path.unlink()

    nodes = []
    initializers = []
    ext_offset = 0
    data_stream = data_path.open("wb")

    def add_external_array(name: str, array: np.ndarray, onnx_dtype: int):
        nonlocal ext_offset
        array = np.ascontiguousarray(array)
        raw = array.tobytes()
        pad = (64 - (ext_offset % 64)) % 64
        if pad:
            data_stream.write(b"\x00" * pad)
            ext_offset += pad
        tensor = TensorProto()
        tensor.name = name
        tensor.data_type = onnx_dtype
        tensor.dims.extend(array.shape)
        tensor.data_location = TensorProto.EXTERNAL
        for k, v in (
            ("location", data_path.name),
            ("offset", str(ext_offset)),
            ("length", str(len(raw))),
        ):
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = v
        data_stream.write(raw)
        ext_offset += len(raw)
        initializers.append(tensor)

    def add_const(name: str, array: np.ndarray):
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(array), name=name))
        return name

    add_const("const_zero_i64", np.array([0], dtype=np.int64))
    add_const("const_neg1_i64", np.array([-1], dtype=np.int64))
    add_const("const_one_i64", np.array([1], dtype=np.int64))
    add_const("const_ple_shape", np.array([1, -1, num_layers, ple_dim], dtype=np.int64))
    add_const("const_ple_slice_shape", np.array([1, -1, ple_dim], dtype=np.int64))
    add_const("const_q_sliding_shape", np.array([1, -1, num_heads, head_dim_sliding], dtype=np.int64))
    add_const("const_kv_sliding_shape", np.array([1, -1, 1, head_dim_sliding], dtype=np.int64))
    add_const("const_attn_sliding_shape", np.array([1, -1, num_heads * head_dim_sliding], dtype=np.int64))
    add_const("const_q_full_shape", np.array([1, -1, num_heads, head_dim_full], dtype=np.int64))
    add_const("const_kv_full_shape", np.array([1, -1, 1, head_dim_full], dtype=np.int64))
    add_const("const_attn_full_shape", np.array([1, -1, num_heads * head_dim_full], dtype=np.int64))
    add_const("const_vnorm_sliding", np.ones((head_dim_sliding,), dtype=np.float32))
    add_const("const_vnorm_full", np.ones((head_dim_full,), dtype=np.float32))
    add_const("const_embed_scale", np.array(math.sqrt(hidden_size), dtype=np.float32))
    add_const("const_ple_embed_scale", np.array(math.sqrt(ple_dim), dtype=np.float32))
    add_const("const_ple_proj_scale", np.array(1.0 / math.sqrt(hidden_size), dtype=np.float32))
    add_const("const_ple_input_scale", np.array(1.0 / math.sqrt(2.0), dtype=np.float32))
    add_const("const_softcap", np.array(softcap, dtype=np.float32))
    if bits == 4:
        add_const("const_u8_mask_low", np.array(0x0F, dtype=np.uint8))
        add_const("const_u8_shift4", np.array(4, dtype=np.uint8))
        add_const("const_f32_eight", np.array(8.0, dtype=np.float32))

    for dim, name in ((head_dim_sliding, "sliding"), (head_dim_full, "full")):
        half = dim // 2
        add_const(f"const_half_{name}", np.array([half], dtype=np.int64))
        add_const(f"const_dim_{name}", np.array([dim], dtype=np.int64))

    with safe_open(str(shared_dir / "model.safetensors"), framework="np") as reader:
        def add_embedding(prefix: str, tag: str, out_name: str, total_dim: int):
            q_weight = reader.get_tensor(f"{prefix}.embedding_quantized")
            scales = reader.get_tensor(f"{prefix}.embedding_scale")[:, :1].astype(np.float32)
            w_name = f"{tag}_emb_q"
            s_name = f"{tag}_emb_scale"
            if bits == 4:
                add_external_array(w_name, q_weight ^ np.uint8(0x88), TensorProto.UINT8)
            else:
                add_external_array(w_name, q_weight.view(np.int8), TensorProto.INT8)
            add_external_array(s_name, scales, TensorProto.FLOAT)

            g_q = f"{tag}_gathered_q"
            g_s = f"{tag}_gathered_s"
            nodes.append(helper.make_node("Gather", [w_name, "input_ids"], [g_q], axis=0))
            nodes.append(helper.make_node("Gather", [s_name, "input_ids"], [g_s], axis=0))
            if bits == 4:
                low_u8 = f"{tag}_low_u8"
                high_u8 = f"{tag}_high_u8"
                low_f = f"{tag}_low_f"
                high_f = f"{tag}_high_f"
                low_s = f"{tag}_low_s"
                high_s = f"{tag}_high_s"
                low_3 = f"{tag}_low_3"
                high_3 = f"{tag}_high_3"
                inter = f"{tag}_inter"
                flat = f"{tag}_flat"
                shape_3 = add_const(f"{tag}_shape3", np.array([1, -1, total_dim // 2, 1], dtype=np.int64))
                shape_flat = add_const(f"{tag}_shapeflat", np.array([1, -1, total_dim], dtype=np.int64))
                nodes.extend([
                    helper.make_node("BitwiseAnd", [g_q, "const_u8_mask_low"], [low_u8]),
                    helper.make_node("BitShift", [g_q, "const_u8_shift4"], [high_u8], direction="RIGHT"),
                    helper.make_node("Cast", [low_u8], [low_f], to=TensorProto.FLOAT),
                    helper.make_node("Cast", [high_u8], [high_f], to=TensorProto.FLOAT),
                    helper.make_node("Sub", [low_f, "const_f32_eight"], [low_s]),
                    helper.make_node("Sub", [high_f, "const_f32_eight"], [high_s]),
                    helper.make_node("Reshape", [low_s, shape_3], [low_3]),
                    helper.make_node("Reshape", [high_s, shape_3], [high_3]),
                    helper.make_node("Concat", [low_3, high_3], [inter], axis=-1),
                    helper.make_node("Reshape", [inter, shape_flat], [flat]),
                    helper.make_node("Mul", [flat, g_s], [out_name], name=f"{tag}_emb_mul"),
                ])
            else:
                cast_f = f"{tag}_cast_f"
                nodes.extend([
                    helper.make_node("Cast", [g_q], [cast_f], to=TensorProto.FLOAT),
                    helper.make_node("Mul", [cast_f, g_s], [out_name], name=f"{tag}_emb_mul"),
                ])

        def add_linear(prefix: str, tag: str, inp: str, out: str, dyn_q=None):
            q_weight = reader.get_tensor(f"{prefix}.weight")
            scales = reader.get_tensor(f"{prefix}.weight_scale").astype(np.float32)
            N = q_weight.shape[0]
            K = q_weight.shape[1] * (2 if bits == 4 else 1)
            w_name = f"{tag}_w"
            s_name = f"{tag}_s"
            if bits == 4:
                bs = group_size if group_size else K
                packed = (q_weight ^ np.uint8(0x88)).reshape(N, K // bs, bs // 2)
                add_external_array(w_name, packed, TensorProto.UINT8)
                add_external_array(s_name, scales.reshape(-1), TensorProto.FLOAT)
                nodes.append(helper.make_node(
                    "MatMulNBits", [inp, w_name, s_name], [out],
                    name=tag, domain="com.microsoft", K=K, N=N, bits=4, block_size=bs, accuracy_level=4,
                ))
            else:
                w_i8 = np.ascontiguousarray(q_weight.view(np.int8).T)
                add_external_array(w_name, w_i8, TensorProto.INT8)
                add_external_array(s_name, scales.reshape(-1), TensorProto.FLOAT)
                if dyn_q is None:
                    q_u8, q_s, q_zp = f"{tag}_a_u8", f"{tag}_a_s", f"{tag}_a_zp"
                    nodes.append(helper.make_node("DynamicQuantizeLinear", [inp], [q_u8, q_s, q_zp]))
                    dyn_q = (q_u8, q_s, q_zp)
                nodes.append(helper.make_node(
                    "MatMulIntegerToFloat",
                    [dyn_q[0], w_name, dyn_q[1], s_name, dyn_q[2]],
                    [out],
                    domain="com.microsoft",
                ))
            return dyn_q

        def add_rmsnorm(weight_key: str, tag: str, inp: str, out: str):
            w = reader.get_tensor(weight_key).astype(np.float32)
            w_name = add_const(f"{tag}_w", w)
            nodes.append(helper.make_node("SimplifiedLayerNormalization", [inp, w_name], [out], axis=-1, epsilon=eps))

        def apply_rope(inp: str, cos_in: str, sin_in: str, tag: str, kind: str, out: str):
            x1 = f"{tag}_x1"
            x2 = f"{tag}_x2"
            nx2 = f"{tag}_nx2"
            rot = f"{tag}_rot"
            p1 = f"{tag}_p1"
            p2 = f"{tag}_p2"
            nodes.extend([
                helper.make_node("Slice", [inp, "const_zero_i64", f"const_half_{kind}", "const_neg1_i64"], [x1]),
                helper.make_node("Slice", [inp, f"const_half_{kind}", f"const_dim_{kind}", "const_neg1_i64"], [x2]),
                helper.make_node("Neg", [x2], [nx2]),
                helper.make_node("Concat", [nx2, x1], [rot], axis=-1),
                helper.make_node("Mul", [inp, cos_in], [p1], name=f"{tag}_mul_cos"),
                helper.make_node("Mul", [rot, sin_in], [p2], name=f"{tag}_mul_sin"),
                helper.make_node("Add", [p1, p2], [out]),
            ])

        # 1. Token embeddings + Per-layer inputs
        add_embedding("model.language_model.embed_tokens", "tok", "raw_embeds", hidden_size)
        nodes.append(helper.make_node("Mul", ["raw_embeds", "const_embed_scale"], ["hidden_0"]))

        add_linear("model.language_model.per_layer_model_projection", "ple_proj", "hidden_0", "ple_proj_raw")
        nodes.extend([
            helper.make_node("Mul", ["ple_proj_raw", "const_ple_proj_scale"], ["ple_proj_scaled"]),
            helper.make_node("Reshape", ["ple_proj_scaled", "const_ple_shape"], ["ple_proj_4d"]),
        ])
        add_rmsnorm("model.language_model.per_layer_projection_norm.weight", "ple_norm", "ple_proj_4d", "ple_proj_normed")

        add_embedding("model.language_model.embed_tokens_per_layer", "ple_tok", "ple_emb_raw", num_layers * ple_dim)
        nodes.extend([
            helper.make_node("Mul", ["ple_emb_raw", "const_ple_embed_scale"], ["ple_emb_scaled"]),
            helper.make_node("Reshape", ["ple_emb_scaled", "const_ple_shape"], ["ple_emb_4d"]),
            helper.make_node("Add", ["ple_proj_normed", "ple_emb_4d"], ["ple_sum"]),
            helper.make_node("Mul", ["ple_sum", "const_ple_input_scale"], ["ple_all"]),
        ])

        donor_sliding = None
        donor_full = None
        for i in range(num_layers):
            is_sliding = layer_types[i] == "sliding_attention"
            if i < num_self_layers:
                if is_sliding:
                    donor_sliding = i
                else:
                    donor_full = i

        for i in range(num_layers):
            p = f"model.language_model.layers.{i}"
            t = f"l{i}"
            h_in = f"hidden_{i}"
            h_out = f"hidden_{i + 1}"
            is_sliding = layer_types[i] == "sliding_attention"
            kind = "sliding" if is_sliding else "full"
            cos_name = "cos_sliding" if is_sliding else "cos_full"
            sin_name = "sin_sliding" if is_sliding else "sin_full"
            mask_name = "mask_sliding" if is_sliding else "mask_full"

            add_rmsnorm(f"{p}.input_layernorm.weight", f"{t}_in_norm", h_in, f"{t}_normed")
            q_dyn = add_linear(f"{p}.self_attn.q_proj", f"{t}_q", f"{t}_normed", f"{t}_q_raw")
            nodes.append(helper.make_node("Reshape", [f"{t}_q_raw", f"const_q_{kind}_shape"], [f"{t}_q_4d"]))
            add_rmsnorm(f"{p}.self_attn.q_norm.weight", f"{t}_q_norm", f"{t}_q_4d", f"{t}_q_normed")
            nodes.append(helper.make_node("Transpose", [f"{t}_q_normed"], [f"{t}_q_t"], perm=[0, 2, 1, 3]))
            apply_rope(f"{t}_q_t", cos_name, sin_name, f"{t}_q_rope", kind, f"{t}_q_rot")

            if i < num_self_layers:
                q_dyn = add_linear(f"{p}.self_attn.k_proj", f"{t}_k", f"{t}_normed", f"{t}_k_raw", dyn_q=q_dyn)
                add_linear(f"{p}.self_attn.v_proj", f"{t}_v", f"{t}_normed", f"{t}_v_raw", dyn_q=q_dyn)
                nodes.extend([
                    helper.make_node("Reshape", [f"{t}_k_raw", f"const_kv_{kind}_shape"], [f"{t}_k_4d"]),
                    helper.make_node("Reshape", [f"{t}_v_raw", f"const_kv_{kind}_shape"], [f"{t}_v_4d"]),
                ])
                add_rmsnorm(f"{p}.self_attn.k_norm.weight", f"{t}_k_norm", f"{t}_k_4d", f"{t}_k_normed")
                nodes.extend([
                    helper.make_node(
                        "SimplifiedLayerNormalization",
                        [f"{t}_v_4d", f"const_vnorm_{kind}"],
                        [f"{t}_v_normed"],
                        axis=-1,
                        epsilon=eps,
                    ),
                    helper.make_node("Transpose", [f"{t}_k_normed"], [f"{t}_k_t"], perm=[0, 2, 1, 3]),
                    helper.make_node("Transpose", [f"{t}_v_normed"], [f"{t}_v_t"], perm=[0, 2, 1, 3]),
                ])
                apply_rope(f"{t}_k_t", cos_name, sin_name, f"{t}_k_rope", kind, f"{t}_k_rot")
                k_curr = f"present_k_{i}"
                v_curr = f"present_v_{i}"
                nodes.extend([
                    helper.make_node("Concat", [f"past_k_{i}", f"{t}_k_rot"], [k_curr], axis=2),
                    helper.make_node("Concat", [f"past_v_{i}", f"{t}_v_t"], [v_curr], axis=2),
                ])
            else:
                donor = donor_sliding if is_sliding else donor_full
                k_curr = f"present_k_{donor}"
                v_curr = f"present_v_{donor}"

            nodes.extend([
                helper.make_node("Transpose", [k_curr], [f"{t}_k_kt"], perm=[0, 1, 3, 2]),
                helper.make_node("MatMul", [f"{t}_q_rot", f"{t}_k_kt"], [f"{t}_scores"]),
                helper.make_node("Add", [f"{t}_scores", mask_name], [f"{t}_masked_scores"]),
                helper.make_node("Softmax", [f"{t}_masked_scores"], [f"{t}_probs"], axis=-1),
                helper.make_node("MatMul", [f"{t}_probs", v_curr], [f"{t}_attn_4d"]),
                helper.make_node("Transpose", [f"{t}_attn_4d"], [f"{t}_attn_t"], perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", [f"{t}_attn_t", f"const_attn_{kind}_shape"], [f"{t}_attn_flat"]),
            ])
            add_linear(f"{p}.self_attn.o_proj", f"{t}_o", f"{t}_attn_flat", f"{t}_o_out")
            add_rmsnorm(f"{p}.post_attention_layernorm.weight", f"{t}_post_attn_norm", f"{t}_o_out", f"{t}_post_attn")
            nodes.append(helper.make_node("Add", [h_in, f"{t}_post_attn"], [f"{t}_res1"]))

            # MLP
            add_rmsnorm(f"{p}.pre_feedforward_layernorm.weight", f"{t}_pre_ffn_norm", f"{t}_res1", f"{t}_ffn_in")
            ffn_dyn = add_linear(f"{p}.mlp.gate_proj", f"{t}_gate", f"{t}_ffn_in", f"{t}_gate_out")
            add_linear(f"{p}.mlp.up_proj", f"{t}_up", f"{t}_ffn_in", f"{t}_up_out", dyn_q=ffn_dyn)
            nodes.extend([
                helper.make_node("Gelu", [f"{t}_gate_out"], [f"{t}_gate_act"], approximate="tanh"),
                helper.make_node("Mul", [f"{t}_gate_act", f"{t}_up_out"], [f"{t}_ffn_mid"], name=f"{t}_ffn_mul"),
            ])
            add_linear(f"{p}.mlp.down_proj", f"{t}_down", f"{t}_ffn_mid", f"{t}_down_out")
            add_rmsnorm(f"{p}.post_feedforward_layernorm.weight", f"{t}_post_ffn_norm", f"{t}_down_out", f"{t}_post_ffn")
            nodes.append(helper.make_node("Add", [f"{t}_res1", f"{t}_post_ffn"], [f"{t}_res2"]))

            # Per-layer input gate
            idx_c = add_const(f"{t}_idx", np.array([i], dtype=np.int64))
            nodes.extend([
                helper.make_node("Gather", ["ple_all", idx_c], [f"{t}_ple_raw"], axis=2),
                helper.make_node("Reshape", [f"{t}_ple_raw", "const_ple_slice_shape"], [f"{t}_ple_slice"]),
            ])
            add_linear(f"{p}.per_layer_input_gate", f"{t}_ple_gate", f"{t}_res2", f"{t}_ple_gate_raw")
            nodes.extend([
                helper.make_node("Gelu", [f"{t}_ple_gate_raw"], [f"{t}_ple_gate_act"], approximate="tanh"),
                helper.make_node("Mul", [f"{t}_ple_gate_act", f"{t}_ple_slice"], [f"{t}_ple_gated"], name=f"{t}_ple_mul"),
            ])
            add_linear(f"{p}.per_layer_projection", f"{t}_ple_proj", f"{t}_ple_gated", f"{t}_ple_proj_out")
            add_rmsnorm(f"{p}.post_per_layer_input_norm.weight", f"{t}_post_ple_norm", f"{t}_ple_proj_out", f"{t}_post_ple")
            nodes.append(helper.make_node("Add", [f"{t}_res2", f"{t}_post_ple"], [f"{t}_res3"]))

            scalar_w = reader.get_tensor(f"{p}.layer_scalar").astype(np.float32)
            scalar_name = add_const(f"{t}_scalar", scalar_w)
            nodes.append(helper.make_node("Mul", [f"{t}_res3", scalar_name], [h_out]))

        # Final norm + last-token slice + lm_head + softcap
        nodes.append(helper.make_node(
            "Slice",
            [f"hidden_{num_layers}", "const_neg1_i64", "const_Max_i64", "const_one_i64"],
            ["hidden_last"],
        ))
        add_const("const_Max_i64", np.array([9223372036854775807], dtype=np.int64))
        add_rmsnorm("model.language_model.norm.weight", "final_norm", "hidden_last", "normed_last")
        add_linear("lm_head", "lm_head", "normed_last", "raw_logits")
        nodes.extend([
            helper.make_node("Div", ["raw_logits", "const_softcap"], ["logits_scaled"]),
            helper.make_node("Tanh", ["logits_scaled"], ["logits_tanh"]),
            helper.make_node("Mul", ["logits_tanh", "const_softcap"], ["logits"]),
        ])

    data_stream.close()

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "T"]),
        helper.make_tensor_value_info("cos_sliding", TensorProto.FLOAT, [1, 1, "T", head_dim_sliding]),
        helper.make_tensor_value_info("sin_sliding", TensorProto.FLOAT, [1, 1, "T", head_dim_sliding]),
        helper.make_tensor_value_info("cos_full", TensorProto.FLOAT, [1, 1, "T", head_dim_full]),
        helper.make_tensor_value_info("sin_full", TensorProto.FLOAT, [1, 1, "T", head_dim_full]),
        helper.make_tensor_value_info("mask_sliding", TensorProto.FLOAT, [1, 1, "T", "KV"]),
        helper.make_tensor_value_info("mask_full", TensorProto.FLOAT, [1, 1, "T", "KV"]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, int(text_cfg["vocab_size"])]),
    ]
    for i in range(num_self_layers):
        dim = head_dim_sliding if layer_types[i] == "sliding_attention" else head_dim_full
        inputs.append(helper.make_tensor_value_info(f"past_k_{i}", TensorProto.FLOAT, [1, 1, "P", dim]))
        inputs.append(helper.make_tensor_value_info(f"past_v_{i}", TensorProto.FLOAT, [1, 1, "P", dim]))
        outputs.append(helper.make_tensor_value_info(f"present_k_{i}", TensorProto.FLOAT, [1, 1, "KV", dim]))
        outputs.append(helper.make_tensor_value_info(f"present_v_{i}", TensorProto.FLOAT, [1, 1, "KV", dim]))

    graph = helper.make_graph(nodes, "gemma4_e2b", inputs, outputs, initializers)
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
    )
    onnx_path.write_bytes(model.SerializeToString())


def precompute_rope_and_masks(shared_dir: Path, max_seq_len: int):
    config = json.loads((shared_dir / "config.json").read_text())["text_config"]
    head_dim_s = int(config["head_dim"])
    head_dim_f = int(config["global_head_dim"])
    theta_s = float(config["rope_parameters"]["sliding_attention"]["rope_theta"])
    theta_f = float(config["rope_parameters"]["full_attention"]["rope_theta"])
    rot_factor_f = float(config["rope_parameters"]["full_attention"].get("partial_rotary_factor", 0.25))
    window = int(config["sliding_window"])

    pos = np.arange(max_seq_len, dtype=np.float32)

    inv_s = 1.0 / (theta_s ** (np.arange(0, head_dim_s, 2, dtype=np.float32) / head_dim_s))
    freqs_s = np.outer(pos, inv_s)
    emb_s = np.concatenate([freqs_s, freqs_s], axis=-1)
    cos_s = np.cos(emb_s).astype(np.float32)[None, None, :, :]
    sin_s = np.sin(emb_s).astype(np.float32)[None, None, :, :]

    rot_dim_f = int(head_dim_f * rot_factor_f)
    inv_f_rot = 1.0 / (theta_f ** (np.arange(0, rot_dim_f, 2, dtype=np.float32) / head_dim_f))
    inv_f = np.concatenate([inv_f_rot, np.zeros(head_dim_f // 2 - len(inv_f_rot), dtype=np.float32)])
    freqs_f = np.outer(pos, inv_f)
    emb_f = np.concatenate([freqs_f, freqs_f], axis=-1)
    cos_f = np.cos(emb_f).astype(np.float32)[None, None, :, :]
    sin_f = np.sin(emb_f).astype(np.float32)[None, None, :, :]

    causal = np.triu(np.full((max_seq_len, max_seq_len), -np.inf, dtype=np.float32), k=1)
    sliding = causal.copy()
    for i in range(max_seq_len):
        start = max(0, i - window + 1)
        if start > 0:
            sliding[i, :start] = -np.inf

    return config, cos_s, sin_s, cos_f, sin_f, sliding, causal


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) not in (2, 3):
        raise SystemExit("usage: onnx_bench.py SHARED_DIR PROTOCOL.json [--export-only]")
    shared_dir = Path(arguments[0]).expanduser().resolve()
    protocol = json.loads(Path(arguments[1]).expanduser().resolve().read_text())
    export_only = len(arguments) == 3 and arguments[2] == "--export-only"

    work_dir = shared_dir.parent
    onnx_path = work_dir / "gemma4-e2b-shared.onnx"
    if not onnx_path.exists() or export_only:
        build_onnx_model(shared_dir, onnx_path)
    if export_only:
        return 0

    import onnxruntime as ort

    mode = protocol.get("mode", "speed")
    warmups = int(protocol.get("warmups", 1))
    context = int(protocol["context_tokens"])
    threads = int(protocol["threads"])

    text_cfg, cos_s, sin_s, cos_f, sin_f, mask_s, mask_f = precompute_rope_and_masks(shared_dir, context)
    num_self_layers = int(text_cfg["num_hidden_layers"]) - int(text_cfg["num_kv_shared_layers"])
    layer_types = list(text_cfg["layer_types"])
    head_dim_s = int(text_cfg["head_dim"])
    head_dim_f = int(text_cfg["global_head_dim"])

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    emit({
        "stage": "runtime",
        "device": "cpu:0",
        "backend": "ONNXRuntime+KleidiAI",
        "kleidiai_enabled": True,
        "threads": threads,
    })

    loading = time.perf_counter()
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    emit({"stage": "load", "ms": (time.perf_counter() - loading) * 1000})

    output_names = ["logits"]
    for i in range(num_self_layers):
        output_names.append(f"present_k_{i}")
        output_names.append(f"present_v_{i}")

    def empty_kv():
        kv = {}
        for i in range(num_self_layers):
            dim = head_dim_s if layer_types[i] == "sliding_attention" else head_dim_f
            kv[f"past_k_{i}"] = np.zeros((1, 1, 0, dim), dtype=np.float32)
            kv[f"past_v_{i}"] = np.zeros((1, 1, 0, dim), dtype=np.float32)
        return kv

    def step(ids_list, start_pos, kv_caches):
        seq_len = len(ids_list)
        kv_len = start_pos + seq_len
        feeds = {
            "input_ids": np.array([ids_list], dtype=np.int64),
            "cos_sliding": cos_s[:, :, start_pos:kv_len, :],
            "sin_sliding": sin_s[:, :, start_pos:kv_len, :],
            "cos_full": cos_f[:, :, start_pos:kv_len, :],
            "sin_full": sin_f[:, :, start_pos:kv_len, :],
            "mask_sliding": mask_s[None, None, start_pos:kv_len, :kv_len],
            "mask_full": mask_f[None, None, start_pos:kv_len, :kv_len],
        }
        feeds.update(kv_caches)
        results = session.run(output_names, feeds)
        logits = results[0].reshape(-1)
        if not np.isfinite(logits).all():
            raise RuntimeError("non-finite ONNX Runtime logits")
        next_kv = {}
        for i in range(num_self_layers):
            next_kv[f"past_k_{i}"] = results[1 + 2 * i]
            next_kv[f"past_v_{i}"] = results[2 + 2 * i]
        return int(np.argmax(logits)), next_kv

    if mode == "eval":
        inputs_list = protocol["input_token_ids_list"]
        max_outputs = int(protocol["max_output_tokens"])

        def run_seq(prompt_ids, limit):
            kv = empty_kv()
            generated = []
            tok, kv = step(prompt_ids, 0, kv)
            for idx in range(1, limit + 1):
                if tok in (1, 106):
                    break
                generated.append(tok)
                if idx == limit:
                    break
                tok, kv = step([tok], len(prompt_ids) + idx - 1, kv)
            return generated

        for _ in range(warmups):
            run_seq(inputs_list[0], min(8, max_outputs))
        all_token_ids = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        eval_started = time.perf_counter()
        for prompt_ids in inputs_list:
            total_prompt_tokens += len(prompt_ids)
            gen = run_seq(prompt_ids, max_outputs)
            total_output_tokens += len(gen)
            all_token_ids.append(gen)
        emit({
            "stage": "eval",
            "items": len(inputs_list),
            "eval_seconds": time.perf_counter() - eval_started,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "token_ids": all_token_ids,
        })
        return 0

    prompt = protocol["input_token_ids"]
    outputs = int(protocol["output_tokens"])
    repeats = int(protocol["repeats"])

    for iteration in range(warmups + repeats):
        kv_caches = empty_kv()

        generated = []
        submitted = time.perf_counter()
        token, kv_caches = step(prompt, 0, kv_caches)
        generated.append(token)
        first = time.perf_counter()
        for idx in range(1, outputs):
            token, kv_caches = step([generated[-1]], len(prompt) + idx - 1, kv_caches)
            generated.append(token)
        last = time.perf_counter()
        prefill = first - submitted
        decode = last - first
        emit({
            "stage": "generation",
            "iteration": iteration,
            "warmup": iteration < warmups,
            "input_token_ids": prompt,
            "token_ids": generated,
            "prompt_tokens": len(prompt),
            "output_tokens": len(generated),
            "decode_tokens": len(generated) - 1,
            "reused_prompt_tokens": 0,
            "first_token_seconds": prefill,
            "decode_seconds": decode,
            "prefill_through_first_token_tps": len(prompt) / prefill,
            "decode_interval_tps": (outputs - 1) / decode,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
