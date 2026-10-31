#!/usr/bin/env python3
"""Compatibility & TPU optimization patcher for `flint_sglang` on `sglang-torchtpu-diffusion:v0.2.0`.

Applies four surgical fixes to installed `flint_sglang/model.py` + `sglang_tpu/moe/router_topk_tpu.py`:
  1. `layer_boundary` migration: replaces removed `LayerCommunicator` / `LayerScatterModes` and
     `get_pp_group()` with `append_stages(declare_attn(), declare_ffn())` and `parallel.pp_group`.
  2. Exact `TP > 1` (`TP=2`, `TP=4`, `TP=8`) all-reduce semantics:
     - Passes `reduce_results=False` to dense `FlintMLP` (layers 0, 1) and `self.shared_expert` (layers 2..48).
     - Sets `self.experts.allow_backend_reduction = False` and `self.experts.reduce_results = False` on
       `FlintSparseMoeBlock` so `(routed_output + shared_output)` is reduced once in `ffn_boundary.finish`
       instead of double-reducing dense/shared experts by `tp_size`!
  3. Runtime toggle `FLINT_TPU_ROUTER_MODE=fused|stock` (default `fused`):
     - `fused`: enables TPU Pallas VMEM `router_topk` (`VALIDATED_NUM_EXPERTS` |= `{256}` +
       `TopK(scoring_func="sigmoid", correction_bias=self.gate.expert_bias, renormalize=True)`).
     - `stock`: preserves reference Python `sigmoid_routing` (`custom_routing_function`) for kernel ablation.
  4. Relaxes CUDA-only `_require_cuda_norm()` guard when `args.device == "tpu"` and accepts Block-FP8
     `.weight_scale_inv` / `.weight_scale` tensors cleanly in `_skipped_by_parent_loader`.
"""

from __future__ import annotations

from pathlib import Path
import site


PATCHED_FLINT_MODEL_PY = '''"""SGLang TPU-compatible implementation of Flint (`RecursiveSI/flint`).

Patched for `sglang-torchtpu-diffusion:v0.2.0` (`layer_boundary` API, exact `TP=1/2/4/8` reduction
across dense SwiGLU + routed/shared MoE experts, Block-FP8 scale coverage, and TPU Pallas `router_topk`).
"""

import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
from flint_sglang.grammar_backend import GRAMMAR_BACKEND_NAME
from sglang.srt.layers.layer_boundary import append_stages, declare_attn, declare_ffn
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import LazyValue, add_prefix
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from torch import nn

logger = logging.getLogger(__name__)

SLIDING_ATTENTION = "sliding_attention"
SUPPORTED_MOE_RUNNERS = frozenset({"auto", "triton", "flashinfer_trtllm_routed"})
FA4_SINK_FIX_MARKERS = ("sink_rescale", "tmem_alloc_warp_ids")
EXPERT_BIAS_SUFFIX = ".mlp.gate.expert_bias"
SINK_SUFFIX = ".self_attn.softmax_offset"


class FlintMLP(Qwen2MoeMLP):
    """Dense or shared-expert SwiGLU MLP that accepts the decoder's optional forward batch."""

    def forward(self, hidden_states: torch.Tensor, forward_batch: object = None) -> torch.Tensor:
        del forward_batch
        return super().forward(hidden_states)


class FlintAttention(Qwen3MoeAttention):
    """Grouped-query attention with QK RMSNorm, learned sinks, optional sliding window and RoPE."""

    def __init__(
        self, *, layer_type: str, skip_rope: bool, sliding_window: int, layer_id: int, prefix: str = "", **kwargs
    ):
        super().__init__(layer_id=layer_id, prefix=prefix, **kwargs)
        self.skip_rope = skip_rope
        # The fused qk-norm/RoPE and fused KV-buffer paths assume RoPE on every layer and no sinks.
        self.use_fused_qk_norm_rope = False
        self.use_tpu_fused_qk_norm_rope = False
        self.compatible_with_fused_kv_buffer = False

        sinks_dtype = torch.float32 if get_server_args().attention_backend == "trtllm_mha" else torch.bfloat16
        self.softmax_offset = nn.Parameter(torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False)

        # SGLang's sliding_window_size excludes the current token; the config value includes it.
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(sliding_window - 1) if layer_type == SLIDING_ATTENTION else -1,
            prefix=add_prefix("attn", prefix),
        )

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        if not self.skip_rope:
            q, k = self.rotary_emb(positions, q, k)
        self._used_fused_qk_norm_rope_last_call = False
        return q, k, v

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        q, k, v, fb = inner_state
        attn_output = self.attn(q, k, v, fb, sinks=self.softmax_offset)
        output, _ = self.o_proj(attn_output)
        return output


class FlintSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """Sparse MoE block with sigmoid top-k routing and an ungated shared expert."""

    def __init__(self, layer_id, config, quant_config=None, prefix=""):
        super().__init__(layer_id, config, quant_config=quant_config, prefix=prefix)

        # Defer TP all-reduce to `ffn_boundary.finish` so `(routed_output + shared_output)` is reduced
        # once across TP ranks without double-reducing `shared_expert`.
        self.experts.reduce_results = False
        self.experts.allow_backend_reduction = False

        # Checkpoints without a router bias leave this at zero.
        self.gate.expert_bias = nn.Parameter(torch.zeros(config.num_experts, dtype=torch.float32), requires_grad=False)

        router_mode = os.environ.get("FLINT_TPU_ROUTER_MODE", "fused").strip().lower()
        if router_mode == "stock":
            def sigmoid_routing(hidden_states, gating_output, topk, renormalize):
                scores = torch.sigmoid(gating_output.float())
                scores_for_choice = scores + self.gate.expert_bias.unsqueeze(0)
                topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
                topk_weights = scores.gather(1, topk_ids)
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
                return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

            self.topk = TopK(
                top_k=config.num_experts_per_tok,
                renormalize=False,
                use_grouped_topk=False,
                layer_id=layer_id,
                custom_routing_function=sigmoid_routing,
            )
        else:
            self.topk = TopK(
                top_k=config.num_experts_per_tok,
                renormalize=True,
                use_grouped_topk=False,
                correction_bias=self.gate.expert_bias,
                scoring_func="sigmoid",
                layer_id=layer_id,
            )

        shared_size = getattr(config, "shared_expert_intermediate_size", 0)
        self.shared_expert = (
            FlintMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
            )
            if shared_size > 0
            else None
        )

    def forward(self, hidden_states, forward_batch=None):
        shared_output = self.shared_expert(hidden_states) if self.shared_expert is not None else None
        output = super().forward(hidden_states, forward_batch)
        if shared_output is not None:
            output = output + shared_output
        return output


class FlintDecoderLayer(Qwen3MoeDecoderLayer):
    """Pre-norm decoder layer: Flint attention followed by a dense MLP or a sparse MoE block."""

    def __init__(self, config, layer_id, start_layer=0, quant_config=None, prefix="", alt_stream=None):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank

        rope_theta, _ = get_rope_config(config)
        self.rope_theta = rope_theta

        self.self_attn = FlintAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=self.rope_theta,
            rope_scaling=None,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            layer_type=config.layer_types[layer_id],
            skip_rope=layer_id in set(config.no_rope_layer_indices),
            sliding_window=config.sliding_window,
        )

        dense_layers = frozenset(config.mlp_only_layers)
        self.is_layer_sparse = layer_id not in dense_layers
        is_next_layer_sparse = (layer_id + 1 < config.num_hidden_layers) and ((layer_id + 1) not in dense_layers)
        if self.is_layer_sparse:
            self.mlp = FlintSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = FlintMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (declare_attn(), self.input_layernorm),
            (
                declare_ffn(
                    sparse=self.is_layer_sparse,
                    next_layer_sparse=is_next_layer_sparse,
                ),
                self.post_attention_layernorm,
            ),
        )

    @property
    def _ffn_summed_by_compute(self) -> bool:
        return False


class FlintModel(Qwen3MoeModel):
    """Embedding, Flint decoder stack and final norm."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix, decoder_layer_type=FlintDecoderLayer)


class FlintForCausalLM(Qwen3MoeForCausalLM):
    """Flint causal language model for SGLang."""

    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        parallel = get_parallel()
        _check_runtime(parallel)
        self.pp_group = parallel.pp_group
        self.config = config
        self.quant_config = quant_config
        self.model = FlintModel(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=parallel.enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        self.routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.model.start_layer, self.model.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, Qwen3MoeSparseMoeBlock)
            }
        )
        self.attn_cp_size = parallel.attn_cp_size
        self.attn_cp_rank = getattr(parallel, "attn_cp_rank", 0)
        self.moe_dp_size = parallel.moe_dp_size
        self.stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        self.expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=config.num_experts,
        )

    def get_attention_sliding_window_size(self) -> int:
        return self.config.sliding_window - 1

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False) -> None:
        """Load checkpoint tensors and fail if any tensor or parameter is left unmatched."""
        params = dict(self.named_parameters())
        loaded_targets: set[str] = set()
        unmatched: list[str] = []
        attn_tp_size = get_parallel().attn_tp_size
        attn_tp_rank = get_parallel().attn_tp_rank
        off_stage_prefixes = tuple(f"{n}." for n, m in self.named_modules() if isinstance(m, PPMissingLayer))

        def canonical_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if name.endswith(SINK_SUFFIX):
                    tensor = _shard_heads(tensor, attn_tp_size=attn_tp_size, attn_tp_rank=attn_tp_rank)
                target = _target_param_name(name, self.stacked_params_mapping, self.expert_params_mapping)
                if target in params:
                    loaded_targets.add(target)
                elif not self._skipped_by_parent_loader(
                    name, target=target, is_mtp=is_mtp, off_stage_prefixes=off_stage_prefixes
                ):
                    unmatched.append(name)
                yield name, tensor

        Qwen3MoeForCausalLM.load_weights(self, canonical_weights(), is_mtp=is_mtp)
        _check_load_coverage(
            params=params,
            loaded_targets=loaded_targets,
            unmatched=unmatched,
            check_missing=not is_mtp and self.quant_config is None,
        )

    def _skipped_by_parent_loader(
        self, name: str, *, target: str, is_mtp: bool, off_stage_prefixes: tuple[str, ...]
    ) -> bool:
        """Checkpoint tensors the stock loader drops or remaps: off-stage/rank, MTP, RoPE buffers, biases, quant scales."""
        layer_id = get_layer_id(name)
        return (
            (layer_id is not None and not self.model.start_layer <= layer_id < self.model.end_layer)
            or name.startswith(off_stage_prefixes)
            or _is_off_rank_expert(name, self.expert_params_mapping)
            or ("mtp" in name) != is_mtp
            or "rotary_emb.inv_freq" in name
            or target.endswith(".bias")
            or (self.quant_config is not None and "scale" in name)
        )


def _check_runtime(parallel) -> None:
    """Fail fast on serving configurations that silently change Flint's outputs; warn on unvalidated ones."""
    args = get_server_args()
    if args.grammar_backend != GRAMMAR_BACKEND_NAME:
        raise RuntimeError(
            f"Flint requires --grammar-backend flint (got {args.grammar_backend}): Flint decides on each turn whether "
            "to reason, and only that backend lets it reason before a constraint or a tool call."
        )
    if getattr(args, "device", "") != "tpu":
        _require_cuda_norm()
    if args.moe_runner_backend not in SUPPORTED_MOE_RUNNERS:
        raise RuntimeError(
            f"--moe-runner-backend {args.moe_runner_backend} bypasses or is not validated with Flint's sigmoid "
            f"routing; use one of {sorted(SUPPORTED_MOE_RUNNERS)}."
        )
    if args.moe_a2a_backend != "none":
        raise RuntimeError(f"--moe-a2a-backend {args.moe_a2a_backend} is not supported for Flint.")
    if "fa4" in (args.attention_backend, args.prefill_attention_backend, args.decode_attention_backend):
        _require_fa4_sink_fix()
    if not args.enable_fp32_lm_head:
        logger.warning("Flint was validated with --enable-fp32-lm-head; outputs without it differ slightly.")
    sizes = {
        "tp": parallel.tp_size,
        "moe_ep": parallel.moe_ep_size,
        "moe_tp": parallel.moe_tp_size,
        "pp": parallel.pp_size,
        "attn_cp": parallel.attn_cp_size,
    }
    if any(size > 1 for size in sizes.values()) or args.enable_dp_attention:
        logger.info("Running Flint on TPU with parallel config: %s", sizes)


def _require_fa4_sink_fix() -> None:
    kernel = Path(sys.modules["sglang"].__file__).parent / "jit_kernel" / "flash_attn" / "cute" / "flash_fwd_sm100.py"
    source = kernel.read_text() if kernel.is_file() else ""
    if not all(marker in source for marker in FA4_SINK_FIX_MARKERS):
        raise RuntimeError(f"--attention-backend fa4 requires fa4-learned-sink-fix.patch (run install.sh): {kernel}")


def _require_cuda_norm() -> None:
    if os.environ.get("FLASHINFER_USE_CUDA_NORM") != "1":
        raise RuntimeError("Flint requires FLASHINFER_USE_CUDA_NORM=1 in the environment that starts SGLang.")
    norm = sys.modules.get("flashinfer.norm")
    if norm is not None and getattr(norm, "_USE_CUDA_NORM", True) is not True:
        raise RuntimeError(
            "flashinfer.norm was imported before FLASHINFER_USE_CUDA_NORM=1 was set; "
            "export it before starting SGLang."
        )


def _shard_heads(tensor: torch.Tensor, *, attn_tp_size: int, attn_tp_rank: int) -> torch.Tensor:
    if tensor.shape[0] % attn_tp_size != 0:
        raise ValueError(f"{tensor.shape[0]} attention heads are not divisible by attention TP size {attn_tp_size}")
    return tensor.chunk(attn_tp_size, dim=0)[attn_tp_rank]


def _target_param_name(
    name: str,
    stacked_params_mapping: list[tuple[str, str, str | int]],
    expert_params_mapping: list[tuple[str, str, int, str]],
) -> str:
    if "mlp.experts" not in name:
        for param_name, weight_name, _ in stacked_params_mapping:
            if weight_name in name:
                return name.replace(weight_name, param_name)
    for param_name, weight_name, _, _ in expert_params_mapping:
        if weight_name in name:
            return name.replace(weight_name, param_name)
    return name


def _is_off_rank_expert(name: str, expert_params_mapping: list[tuple[str, str, int, str]]) -> bool:
    return "mlp.experts" in name and any(weight_name in name for _, weight_name, _, _ in expert_params_mapping)


def _check_load_coverage(
    *, params: dict[str, nn.Parameter], loaded_targets: set[str], unmatched: list[str], check_missing: bool
) -> None:
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} checkpoint tensors match no Flint parameter, e.g. {unmatched[:5]}")
    if not check_missing:
        return
    missing = sorted(name for name in params if name not in loaded_targets and not name.endswith(EXPERT_BIAS_SUFFIX))
    if missing:
        raise RuntimeError(f"{len(missing)} Flint parameters were not loaded from the checkpoint, e.g. {missing[:5]}")
    logger.info("Flint weights loaded: %d parameters matched the checkpoint.", len(loaded_targets))


EntryClass = FlintForCausalLM
'''


def apply_flint_tpu_patches() -> None:
    sp_dirs = [Path(p) for p in site.getsitepackages()]
    sp = next((p for p in sp_dirs if (p / "flint_sglang" / "model.py").exists()), None)
    if sp is None:
        sp = Path("/usr/local/lib/python3.12/site-packages")

    flint_model_py = sp / "flint_sglang" / "model.py"
    if not flint_model_py.exists():
        raise FileNotFoundError(f"Missing {flint_model_py} — install flint_sglang wheel first!")
    flint_model_py.write_text(PATCHED_FLINT_MODEL_PY)
    print(f"[patch-flint] Patched {flint_model_py} (layer_boundary + exact TP=1/2/4/8 reduction + router toggle)", flush=True)

    rtopk_py = sp / "sglang_tpu" / "moe" / "router_topk_tpu.py"
    if rtopk_py.exists():
        rtxt = rtopk_py.read_text()
        old = "VALIDATED_NUM_EXPERTS = frozenset({288, 896})"
        new = "VALIDATED_NUM_EXPERTS = frozenset({256, 288, 896})"
        if old in rtxt:
            rtopk_py.write_text(rtxt.replace(old, new, 1))
            print(f"[patch-flint] Enabled 256-expert Pallas router_topk in {rtopk_py}", flush=True)
        elif new in rtxt:
            print(f"[patch-flint] 256-expert Pallas router_topk already active in {rtopk_py}", flush=True)


if __name__ == "__main__":
    apply_flint_tpu_patches()
