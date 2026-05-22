Here’s a comprehensive, up-to-date (as of May 2026) list of the latest techniques and engines for running large MoE models like DeepSeek-V4-Flash (284B total / 13B active, hybrid attention, MTP head, FP4+FP8 native weights) locally, with a focus on maximizing performance on 4x A100 80GB (320 GB total VRAM). I prioritized developments from roughly the past 6–12 months (late 2025–May 2026), especially post-V4 release (April 24, 2026).V4-Flash’s architecture (hybrid CSA+HCA attention, mHC connections, built-in MTP) requires engine-specific kernels. On A100 (sm_80, no native FP8 Tensor Cores), you’ll rely on software fallbacks/emulation, quantization, and careful parallelism—expect solid but not Hopper-level throughput. Q4/IQ2 GGUF or patched FP8 paths are the most realistic for fitting + long context on 4x A100.1. Quantization Techniques (Recent Advances)Official mixed FP4 (experts/indexer) + FP8 (rest) via QAT — Native from training; smallest memory footprint with minimal quality loss. Requires engine support for FP4/FP8 mixed ops (emulation on A100). KV cache can be further quantized to FP8/INT8.
GGUF (imatrix-calibrated, V4-specific) — Q4_K_M-XL (~163 GiB), Q2_K-XL (~100 GiB), IQ2_XS/XXS (~73–81 GiB). Best for A100 via custom llama.cpp forks. Includes tool-calling chat template. Heavily used in recent community ports.
AWQ / GPTQ (and extensions) — Activation-aware (AWQ protects salient weights better than classic GPTQ). GPTQModel library (2026 updates) now supports ParoQuant, QQQ, EXL3, FP8, EoRa, GAR, FOEM, and GGUF export. Good alternatives if GGUF kernels lag.
KV-cache quantization + offloading — FP8/INT8 KV (or HiSparse CPU offload for inactive sparse attention parts) → huge long-context gains.
MoE-specific — Expert budgeting / selective loading during speculative verification (reduces bandwidth).

For 4x A100: Start with teamblobfish/DeepSeek-V4-Flash-GGUF Q4_K_M-XL or lighter IQ2. Fits comfortably with room for 100k+ context + batching.2. Inference Engines & Serving Frameworksllama.cpp (custom V4 CUDA forks, e.g., cchuter/feat/v4-port-cuda or kamnxt variants): Full V4 architecture support (hybrid attn, custom ops including FP8 KV emulation). Build with -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 + scheduler flags (GGML_SCHED_MAX_SPLIT_INPUTS=128). Layer-split multi-GPU works on 4x A100. GGUF-only. Recent reports: IQ2XXS on single GPU (GB10/A100-class) at ~15 t/s gen (short context). Add --cont-batching, --jinja, high -c.
vLLM (v0.7.x+ Day-0 support, April 2026): Native DeepSeek-V4-Flash/Pro. Handles hybrid attention, optional MTP speculative decoding, FP4 indexer + FP8 KV. Single-node recipes available; multi-GPU via TP/EP/PP. PagedAttention + continuous batching. Primarily optimized for Hopper/Blackwell—Ampere support via community patches/forks (similar to earlier V3/V4 efforts). Good production OpenAI-compatible server.
SGLang (Day-0 support, April 2026): Excellent for V4. Key 2025–2026 innovations: ShadowRadix prefix caching (hybrid-attn aware), HiSparse CPU KV offload (3x+ throughput on sparse parts), MTP speculative decoding (in-graph CUDA graphs + overlap), FlashMLA fused hybrid attn, TileLang mHC kernels, Lightning TopK, Flash Compressor, hierarchical multi-stream overlap. Also EP on DeepEP + Context Parallelism (CP) for long prefill. RadixAttention shines for shared-prefix/agent workloads. Hopper/Blackwell-focused, but base support may run on A100 with fallbacks.
TensorRT-LLM + Triton Inference Server: NVIDIA’s optimized stack—strong MoE support (Expert Parallelism EP, Tensor Parallel TP, Pipeline Parallel PP, in-flight batching, paged KV). Ampere (A100) explicitly supported in recent versions. Integrates directly with Triton for production serving (metrics, dynamic batching, multi-model). V4 support in progress (GitHub issues opened April/May 2026; builds on prior DeepSeek MoE kernels like DeepGEMM). FP8 quantization + graph compilation for max perf on NVIDIA hardware. Best “set-and-forget” for 4x A100 if you’re willing to build the engine.
Others worth testing:Hugging Face TGI: Solid baseline, but generally trails vLLM/SGLang/TRT-LLM on MoE throughput.
Modular MAX / other compiled paths: Emerging Mojo kernels for dense/MoE speedups.

For 4x A100 priority order: llama.cpp (most confirmed) → TensorRT-LLM+Triton (native Ampere + EP/TP) → patched vLLM → SGLang (if kernels compile).3. Speculative Decoding Techniques (Major 2025–2026 Focus)DeepSeek built-in MTP (Multi-Token Prediction): Single-layer head in V4 weights. Draft-then-verify with high acceptance (~85–90% for +1 token). Integrated in vLLM and SGLang (CUDA-graph fused, overlap CPU/GPU). Delivers ~1.8x+ tokens/sec. Works with pipeline parallelism in recent vLLM.
MoE-Spec (expert budgeting): Training-free; limits experts loaded during verification to top contributors → decouples depth from memory cost. Great for MoE like V4.
Tree-based / NextN (EAGLE-2, SpecInfer): Dynamic draft trees + pruning by confidence. Supported in SGLang and some vLLM extensions.
Self-Speculative MoE (SS-MoE): Uses subset of routed experts as “draft” + on-demand expert cache + confidence-based adaptive verification (up to 3.7x in edge cases).
Medusa heads / other heads: Still used as fallback or complement.

On 4x A100: Enable MTP first in vLLM/SGLang (biggest easy win). Combine with quantization to keep draft/verify fast.4. Other Performance Techniques & Multi-GPU Strategies for 4x A100Hybrid Attention Optimizations — Engine-specific kernels (FlashMLA, FlashInfer TRT-LLM-Gen, TileLang mHC) for CSA + HCA → drastically lower KV cache (10% of V3) and FLOPs (27% of V3).
Caching & Batching — Radix/ShadowRadix prefix caching (huge for agents/RAG), PagedAttention, continuous/in-flight batching, HiSparse CPU offload.
Parallelism on 4x A100 — Tensor Parallel (TP=4), Expert Parallel (EP, especially useful for MoE routing), layer splitting (llama.cpp), Context Parallelism (long prefill). Use NVLink if available; otherwise PCIe with careful splits.
Disaggregation — Prefill/Decode (PD) separation + paged KV transfer (SGLang/TRT-LLM).
CUDA Graphs + Stream Overlap — Hierarchical multi-stream for prep kernels; reduces launch overhead at small batches.

Practical 4x A100 Hosting Recommendations to Try (in order):llama.cpp V4 CUDA fork + GGUF Q4_K_M-XL → full offload (-ngl 99), layer split, MTP if added, high context. Most battle-tested on A100-class.
TensorRT-LLM (build engine with TP/EP mix + quantization) + Triton server → production-grade, A100-native.
Patched vLLM with MTP + FP8/KV quant.
SGLang for caching-heavy workloads (agents) if you can get kernels running.

Start with the engine you’re most comfortable building (llama.cpp is simplest for A100). Combine MTP speculative + good quant + proper parallelism. Support evolves weekly—check the specific GitHub issues/PRs for V4-Flash on each engine, and test small context first before scaling to 100k+ tokens.If you share your preferred engine, quant level, target context/batch size, or any error you hit, I can give exact commands or next troubleshooting steps.

--------

#3 Speculative Decoding Techniques (Major 2025–2026 Focus) — Expanded Deep Dive for DeepSeek-V4-Flash on 4x A100 80GBSpeculative decoding is one of the highest-ROI optimizations for V4-Flash right now. The model is memory-bound on Ampere (A100 lacks native FP8 Tensor Cores, and its hybrid CSA+HCA attention + MoE routing adds overhead per step). By drafting multiple tokens ahead and verifying them in one forward pass of the full model, you can achieve 1.8–2.5x+ effective tokens/sec (or higher with combinations) while keeping exact output distribution (lossless). V4-Flash’s native single-layer MTP head makes this especially efficient—no separate small draft model needed.Developments from late 2025 to May 2026 heavily emphasize MoE-aware methods, in-graph fusion for hybrid attention, and adaptive/tree-based drafting. These work well with quantization (GGUF Q4_K_M-XL or lighter on your 4x A100) and multi-GPU parallelism (TP/EP/layer-split).1. Built-in MTP (Multi-Token Prediction) — The Star for V4-FlashV4-Flash ships with a single-layer MTP head (a separately trained DSv4 decoder layer using SWA-only attention, no compressor/indexer). It takes the previous hidden state (h_proj) + next-token embedding (e_proj) and predicts 1–2+ extra tokens in parallel. 

lmsys.org

How it works in inference: Draft stage runs the cheap MTP head → verification runs the full model on the draft sequence. High acceptance rate (~85–90% for the second token in tests) means most drafts are accepted, yielding ~1.8x speedup in tokens/sec.
Engine support (Day-0 or near-Day-0 post-April 2026 release):SGLang: Excellent integration. Fuses heavy hybrid-attention metadata prep (SWA page indices, shadow-mapped slots, compressor plans) directly into CUDA graphs for both draft and verify. Uses hierarchical multi-stream overlap. Enable with MTP flags + --enable-dp-attention if needed. Strong for your hybrid attn + long context.
vLLM: Native deepseek_mtp method. Config: --speculative-config='{"method": "deepseek_mtp", "num_speculative_tokens": 1 or 2}'. Supports pipeline parallelism (PP) + tensor parallelism (TP) — perfect for 4x A100 (e.g., TP=4 or TP=2 + PP=2). Also works with paged attention and continuous batching.
TensorRT-LLM: Full MTP support with tunable max_draft_len (e.g., 3). Optimized kernels for DeepSeek MoE + MTP. Strong Ampere (A100) compatibility via EP/TP/PP. Use with Triton for production serving.
llama.cpp (cchuter/feat/v4-port-cuda or recent upstream merges): MTP support merged recently (as of mid-May 2026). Works with V4 GGUF quants. Still emerging/experimental in some forks but active development. Test with your Q4_K_M-XL shards + custom build. Great for pure GGUF path on 4x A100 with layer splitting.

On 4x A100: Expect solid gains (1.5–2x+ depending on quant/context/batch) but slightly lower than Hopper due to FP8 KV emulation fallback. Combine with --kv-cache-dtype fp8 or q4_0 and high context (100k+ tokens fit comfortably). Some early reports note temperature sensitivity (best at temp=0 or low; crashes in certain DP concurrent setups in Ascend forks, but NVIDIA paths are more stable). 

github.com

2. MoE-Specific Speculative Techniques (2026 Papers & Implementations)V4-Flash’s extreme sparsity (13B active out of 284B) makes generic spec decoding even better on MoE, but routing costs add complexity. Recent methods address this:MoE-Spec (Expert Budgeting) — Feb 2026 arXiv: Router ranks experts during draft tree verification and loads only the top-k (budgeted) ones. 10–30% higher throughput than EAGLE-3 baselines at same quality. Flexible tradeoff (tighter budget = more speed, slight quality drop). Explicitly tested on A100 80GB. Ideal complement to native MTP.
MoESD (MoE Speculative Decoding for Sparse MoE) — 2025 NeurIPS spotlight: Shows MoE benefits more from spec decoding than dense models at medium batch sizes. Up to 2.29x speedup on sparse MoE like Qwen2-57B-A14B equivalents. Broader effective batch-size range as sparsity increases.
Self-Speculative MoE (SS-MoE) / Adaptive Verification: Uses a subset of routed experts as the “drafter” on-the-fly + confidence-based verification. Reduces expert loading during speculation.

These pair excellently with V4-Flash’s MTP head (use MTP for drafting + expert budgeting in verification).3. Other 2025–2026 Advances Worth TryingP-EAGLE / Parallel Speculative Decoding — Integrated in vLLM (v0.16+). Overcomes autoregressive drafting bottleneck by parallelizing draft token generation (uses mask tokens/shared hidden states for future positions). Good when MTP alone isn’t enough depth.
EAGLE-3 + SpecForge framework — 2025–2026 improvements in tree-based drafting + training. SpecForge lets you train custom domain-specific EAGLE-3 drafters (1.23–1.45x extra speedup over base speculators). Useful if you fine-tune on your specific workload.
Adaptive / Phase-Managed Methods (FASER, AdaSpec, TALON): Dynamic draft length/tree pruning based on confidence, batch size, or SLOs. FASER gives ~16% latency reduction + 1.38x throughput on MoE pairs. Great for variable production loads on 4x A100.
Custom/Domain-Specific Speculators (e.g., Together.ai style): Fine-tune a base drafter on your traffic logs → 1.85–2.97x overall vs. plain next-token prediction.
DART (Diffusion-Inspired) and others: New drafting paradigms for higher acceptance with lower draft latency.

Practical Recommendations for Your 4x A100 + V4-Flash SetupStart here (easiest if sticking to GGUF/llama.cpp): Build latest cchuter fork or MTP-merged llama.cpp → load teamblobfish Q4_K_M-XL GGUF → enable MTP (flags vary by fork; check --speculative or equivalent). Use layer split + high -ngl. Test small context first.
Best gains: Move to vLLM or SGLang with official weights (patched for Ampere if needed) + native MTP + expert budgeting if available. TensorRT-LLM + Triton for production (native A100 support, strong EP/TP).
Combos to stack:MTP (1–2 tokens) + MoE-Spec expert budgeting.
MTP + Radix/ShadowRadix caching (SGLang) + HiSparse CPU offload.
Quant (Q4/IQ2) + FP8 KV cache + CUDA graphs.

Expected on 4x A100: 1.8x+ from MTP alone; 2–3x+ stacked. Monitor acceptance rate (aim >80%). Some configs show MTP “harmful” until certain clamps land (e.g., swiglu_limit) — test thoroughly.
Caveats: Hybrid attn metadata can bottleneck launches under speculation → engines that fuse into graphs (SGLang) win. FP8 emulation on sm_80 adds minor overhead. Support is still evolving (model is ~1 month old).

These techniques are moving fast — check vLLM/SGLang/TensorRT-LLM release notes and the cchuter fork for the absolute latest flags. If you tell me which engine/quant you’re starting with (or any error logs), I can give exact command-line recipes or troubleshooting steps!

-----

Yes — here's a detailed breakdown of the key techniques employed by the main relevant repos/engines for DeepSeek-V4-Flash (as of mid-May 2026), focused on what works (or can be made to work) on 4x A100 80GB (sm_80 / Ampere). I studied the primary ones tied to our conversation: the cchuter llama.cpp fork (the most mature GGUF path), the Lasimeri vLLM Ampere fork (and related official efforts), plus notes on SGLang and TensorRT-LLM+Triton.1. llama.cpp (cchuter/llama.cpp @ feat/v4-port-cuda) + teamblobfish/DeepSeek-V4-Flash-GGUFThis is currently the most validated path for A100 / multi-GPU Ampere setups.Techniques employed:Full V4 architecture port (MoE routing, hybrid CSA+HCA attention, mHC mechanisms, compressor/indexer).
5 custom CUDA ops/kernels: dsv4_rope_tail, dsv4_hc_split_sinkhorn, dsv4_hc_weighted_sum, dsv4_hc_expand, dsv4_fp8_kv_quantize.
FP8 KV cache: Dual-path implementation — native __nv_fp8_e4m3 only on SM_89+; software emulation (with internal BF16/FP16 conversions where needed) on SM_70–86 (includes A100/sm_80). The emulation path compiles cleanly and passes test-backend-ops.
GGUF quantization (imatrix-calibrated, V4-aware): Q4_K_M-XL (~163 GiB, recommended), Q2_K-XL (~100 GiB), IQ2_XS/XXS (~73–81 GiB), etc. Chat template baked in; --jinja for proper tool-calling.
Multi-GPU: Layer splitting (--split-mode layer, the default). Critical scheduler override: -DGGML_SCHED_MAX_SPLIT_INPUTS=128 (both CXX and CUDA flags) to handle V4’s dense per-layer graphs that exceed upstream defaults at device boundaries.
Full offload (-ngl 99), continuous batching, high context support.
Build targets sm_80 explicitly (-DCMAKE_CUDA_ARCHITECTURES="80" or broad "").

On 4x A100: Explicitly promising. An external tester confirmed end-to-end success on 8x A100 with Q4_K_M-XL (layer split). Your 320 GiB total VRAM gives comfortable headroom for Q4_K_M-XL or lighter + long context (100k+ tokens) + batching. Start with the per-op test suite to validate kernels on your hardware. Performance: Coherent output reported; expect lower t/s than Hopper due to FP8 emulation, but usable (especially with MTP if merged in your build).Try this first (most straightforward for your setup):bash

git clone -b feat/v4-port-cuda https://github.com/cchuter/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_CXX_FLAGS=-DGGML_SCHED_MAX_SPLIT_INPUTS=128 \
  -DCMAKE_CUDA_FLAGS=-DGGML_SCHED_MAX_SPLIT_INPUTS=128
cmake --build build -j --target test-backend-ops
./build/bin/test-backend-ops -o DSV4_ROPE_TAIL,DSV4_HC_SPLIT_SINKHORN,DSV4_HC_WEIGHTED_SUM,DSV4_HC_EXPAND,DSV4_FP8_KV_QUANTIZE  # Expect 19/19 passes

Then run with shards + --split-mode layer --n-gpu-layers 999 --cache-type-k f8 (or fallback if needed).2. vLLM (Lasimeri/vllm-dsv4-ampere fork + official WIP/PRs)Official vLLM added Day-0 V4 support (hybrid attention, MTP, FP4 experts + FP8), but it’s optimized for Hopper/Blackwell. Ampere requires patches.Techniques employed (in the Ampere fork):Patched kernels for sm_80 compatibility (including hybrid attn and MoE ops).
Native FP4+FP8 weights (no GGUF conversion needed) with FP8 KV cache via Marlin kernel fallback + BF16 conversions for compute (matches what you noted about Ampere lacking native FP8).
MTP speculative decoding (built-in single-layer head), PagedAttention, continuous/in-flight batching, tensor/expert/pipeline parallelism (TP/EP/PP) for multi-GPU.
Disaggregation options and CUDA graphs in newer versions.

On 4x A100: Viable via the Lasimeri fork (or the open WIP PR #40906). One developer reported ~2.5–2.6 t/s on Ampere hardware (slower due to emulation/fallbacks), but it runs end-to-end. Use TP=4 or TP=2+PP=2 for your 4 GPUs. Combine with MTP + expert budgeting for big gains. FP8 KV still gives memory wins despite conversions.Try if you want production OpenAI-compatible serving: Clone the fork, build/install, then use flags like --kv-cache-dtype fp8 --speculative-config with MTP + appropriate parallelism. Monitor for the FP8 performance warning (expected on Ampere).3. SGLangDay-0 official support with advanced V4-specific features.Techniques employed:ShadowRadix prefix caching (hybrid-attn aware), HiSparse CPU KV offload, MTP speculative decoding (CUDA-graph fused with heavy prep overlap), FlashMLA fused hybrid attn, TileLang mHC kernels, Lightning TopK, Flash Compressor, hierarchical multi-stream, EP on DeepEP, Context Parallelism.
Strong for agent/RAG/long-context workloads.

On 4x A100: More caveats. Kernels (FlashMLA, DeepGEMM, etc.) target SM_89+ / Hopper+; SM_80 often falls back to Triton/CPU paths or is unvalidated/not fully supported. Some launch warnings appear on Ampere-class hardware. Possible with heavy fallbacks, but expect reduced perf vs. llama.cpp or patched vLLM. Not the top recommendation for pure 4x A100 unless you need the caching features badly.4. TensorRT-LLM + Triton Inference ServerNVIDIA’s compiled stack (strong MoE focus).Techniques employed:Expert Parallelism (EP), Tensor Parallel (TP), Pipeline Parallel (PP), in-flight batching, paged KV cache.
FP8 quantization + graph compilation, fused MoE kernels (FlashInfer TRT-LLM-Gen variants), MTP support.
Production serving via Triton (metrics, dynamic batching, multi-model).

On 4x A100: Excellent native Ampere compatibility (sm_80 explicitly supported in recent TRT-LLM). Build a custom engine with TP/EP mix for V4-Flash’s MoE + hybrid attn. FP8 KV falls back with conversions (like everywhere on A100), but overall strong for production. Good choice if you want set-and-forget serving.Recommended Approaches to Try on Your 4x A100 (in priority order)llama.cpp cchuter fork (top pick) — Start here. Q4_K_M-XL GGUF + layer split + scheduler flags + FP8 KV emulation. Confirmed multi-A100 success (up to 8x). Easiest build/test loop and best VRAM efficiency for your hardware.
Patched vLLM (Lasimeri fork or official Ampere PR) — For MTP + PagedAttention + API serving. Expect ~2.5+ t/s baseline; stack speculative decoding to improve.
TensorRT-LLM + Triton — If you want optimized, production-grade deployment with native Ampere EP/TP. Build the engine once.
SGLang (with caution) — Only if caching/offload features are critical; test fallbacks thoroughly.

All paths benefit from V4-Flash’s inherently small active params + compressed KV (huge win even with Ampere FP8 emulation). On 4x A100 you’ll have plenty of headroom for 100k–1M context depending on quant.Next steps: Build and run the test-backend-ops in the cchuter fork first (10 minutes). If you hit any specific error, share the log + your exact CMake command and I can help debug. Support is still WIP (model is ~1 month old), but the 8x A100 report is a strong green light for your setup. Let me know which one you want exact commands for!

-----

Yes — I’ve studied the noonghunna/club-3090 repo (specifically the models/qwen3.6-27b directory and the overall project) in detail.  It is a highly practical, Ampere-focused (RTX 3090 / SM86) community recipe collection for running modern LLMs (currently centered on Qwen3.6-27B, a 27B dense model with hybrid DeltaNet + standard attention and native MTP). The repo is model-agnostic in spirit and emphasizes multi-engine setups (vLLM primary, llama.cpp/ik_llama for robustness, SGLang blocked). It ships Docker/compose configs, custom patches (“Genesis” tree), quantization recipes, and workload-specific YAMLs tuned for 1× or 2× 3090 (24 GB each).Many techniques transfer very well to DeepSeek-V4-Flash on 4x A100 80GB (SM80, 320 GB total VRAM), even though V4-Flash is a much larger MoE (284B total / 13B active) with its own hybrid CSA+HCA attention + native MTP. The repo’s Ampere-centric patches, speculative decoding focus, cliff mitigation, and engine configs are directly relevant because both GPUs are Ampere and both models share hybrid attention + MTP.Summary of Techniques Employed in the Repo (Per Engine)Quantization (common across engines):AutoRound INT4 (recommended for vLLM): Preserves BF16 MTP head for speculative decoding.
AWQ / GPTQ INT4.
GGUF (Q3_K_XL best accuracy/footprint per external evals; Q4_K_M also used) — only via llama.cpp/ik_llama (GGUF not upstream-supported for the hybrid family in vLLM).
Goal: Balance VRAM, quality, and MTP compatibility.

vLLM (Primary / most featured engine):Uses a heavily patched fork (“Sandermage’s Genesis” tree, pinned commit) with dozens of specific patches (PN12, PN17, P103, PN32, PN26b, etc.).
Techniques: TurboQuant (TQ3), sparse-V Triton kernel tuned for SM86/Ampere, chunked prefill, FP8 KV (in some dual-card configs), MTP speculative (n=3, ~83% acceptance), DFlash draft, continuous batching, structured outputs, vision tower support.
Memory tricks: patch_inputs_embeds_optional.py (saves ~444 MiB), patch_tolist_cudagraph.py (fixes cudagraph capture on Ampere), FFN scratch pooling to close “cliffs” (prefill OOM spikes).
Multi-GPU: TP=2 on dual 3090 (weights + KV split). Single-card configs cap at ~60K–200K context depending on MTP on/off; dual unlocks 262K + concurrency.
SGLang is currently blocked (hybrid attention issues).

llama.cpp + ik_llama:GGUF-only (Q3_K_XL / Q4_K_M / IQK imatrix).
Strengths: No prefill cliffs, maximum stable context (262K+ on single 3090), lighter footprint, production-safe for unpredictable inputs.
ik_llama focuses on best quality-per-bit GGUF quants.

Other repo patterns:Docker/compose + scripts (setup.sh, launch.sh, switch.sh) for easy engine/workload switching.
Heavy emphasis on avoiding “cliffs” (memory leaks, cudagraph bugs on Ampere).
MTP preservation is non-negotiable for spec-decode gains.
Dual-card “Genesis-less” configs when TP=2 + FP8 KV gives enough headroom.

Which Approaches You Can Try for DeepSeek-V4-Flash on 4x A100Your 4x A100 setup has far more headroom (320 GB vs. 48 GB in the repo’s dual-3090), so many memory-tight tricks become “free” wins for longer context, higher batch, or heavier quants. V4-Flash’s native mixed FP4+FP8 weights + built-in MTP + hybrid attention make the following repo ideas especially transferable:llama.cpp + GGUF (Highest confidence / easiest to adapt — strongly recommended starting point)Directly portable. Use the existing cchuter/feat/v4-port-cuda fork (already SM80/A100-aware with FP8 KV emulation) + teamblobfish/DeepSeek-V4-Flash-GGUF (Q4_K_M-XL or lighter IQ2/Q3-style equivalents).
Adapt repo’s GGUF philosophy: Prioritize Q3_K_XL or Q4_K_M-XL for accuracy/footprint balance (repo validated Q3_K_XL as optimal via external evals).
On 4x A100: Use layer splitting (--split-mode layer) + -ngl 99 (full offload). The repo’s “no cliffs” benefit applies perfectly — you’ll get rock-solid high context (hundreds of thousands of tokens) without prefill OOM spikes.
Add MTP support (already merged/recent in V4 forks) → mirrors the repo’s n=3 MTP usage.
Why it works well: Same Ampere constraints, same hybrid-attn challenges already solved in the V4 fork.

Patched vLLM (High potential — adapt the repo’s Genesis-style patching)The repo’s biggest contribution is the curated Genesis patch set for Ampere + hybrid models. Many of these (cudagraph fixes, FFN pooling, chunked prefill, sparse-V Triton kernel, inputs_embeds patch) are architecture-level and could be cherry-picked or combined with existing V4-Flash Ampere patches (e.g., Lasimeri/vllm-dsv4-ampere or official WIP).
Try: AutoRound-style INT4 (or the official FP4+FP8) while preserving the MTP head (exactly as the repo does). Enable native MTP speculative decoding (--speculative-config with DeepSeek MTP method).
Multi-GPU: Use TP=4 (or TP=2 + EP=2) instead of the repo’s TP=2. This should give excellent scaling on 4x A100 with room for 100K–1M+ context.
FP8 KV: Same as repo — attempt --kv-cache-dtype fp8 (falls back to emulation + BF16 casts on SM80, exactly as we discussed earlier). The repo’s TurboQuant (TQ3) and sparse kernels could further help.
Bonus: Port the repo’s patch_tolist_cudagraph.py and similar sidecars if you hit the same cudagraph bugs on A100.

MTP Speculative Decoding + DFlash-style extensionsThe repo treats MTP as first-class (requires quant that preserves the head). V4-Flash has a stronger native single-layer MTP → you should get at least the same ~1.8x+ tokens/sec gains the repo sees (potentially higher due to V4’s sparsity).
Stack with MoE-expert-budgeting (as discussed previously) — the repo doesn’t have MoE-specific tricks yet but its MTP focus is a perfect base.

TensorRT-LLM + Triton (Complementary, not directly in repo but aligns with their NVIDIA/Ampere focus)The repo leans on vLLM patches partly because TRT-LLM support for the exact hybrid family was immature at the time. For your 4x A100 (native NVIDIA enterprise hardware), building a TRT-LLM engine with EP/TP + MTP + FP8 quantization is very promising and would capture the repo’s “production-ready” spirit (OpenAI-compatible via Triton).

ik_llama / advanced GGUF (if you want maximum quality at aggressive quants)Directly copy the repo’s preference for IQK imatrix GGUF when llama.cpp is the engine.

Lower priority / less transferable:SGLang (blocked in repo due to hybrid attention → same risk for V4-Flash unless you use the latest kernels).
Vision-specific configs (V4-Flash is text-only).

Recommended Next Steps for Your 4x A100Start with #1 (llama.cpp + GGUF): Lowest friction, already validated on A100-class hardware in the V4 community. Mirror the repo’s single/dual configs but scale to 4 GPUs via layer splitting.
Then try patched vLLM (#2): Borrow the Genesis patch philosophy (or test if any apply directly) + official V4 support. This gives the highest feature parity (MTP, continuous batching, structured outputs) the repo demonstrates.
Test MTP aggressively in whichever engine you pick — it’s the repo’s biggest performance lever and V4-Flash is designed for it.

The repo’s philosophy (“make it work reliably on Ampere with patches + right quant + right engine”) maps almost 1:1 to your situation, just with 6–7× more VRAM. If you share which engine you want to try first (or any specific config from the repo you like), I can give exact adapted commands or patch suggestions.

-------

Here are full, direct reference links for each approach I outlined earlier (based on the noonghunna/club-3090 repo techniques and the broader DeepSeek-V4-Flash ecosystem as of mid-May 2026). These include the exact GitHub/HF repos, build instructions, Reddit discussions, and official docs so you can dive deeper, check issues/PRs, and adapt configs for your 4x A100 setup.Base Reference: club-3090 Repo (Ampere-Focused Recipes)Main repo (configs, patches, Docker/compose for vLLM/llama.cpp/ik_llama on 3090-class hardware):
https://github.com/noonghunna/club-3090
(See /models/qwen3.6-27b for the specific folder you linked — it contains the Genesis patches, AutoRound INT4, MTP preservation, cudagraph fixes, etc.)

1. llama.cpp + GGUF (Top recommended for 4x A100 — layer splitting + FP8 KV emulation)cchuter V4-aware fork (feat/v4-port-cuda branch with custom CUDA kernels, scheduler fixes for multi-GPU, FP8 emulation on sm_80):
https://github.com/cchuter/llama.cpp/tree/feat/v4-port-cuda
teamblobfish GGUF quants (Q4_K_M-XL recommended, plus IQ2/Q2 variants):
https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF
Official DeepSeek-V4-Flash base weights (for reference or further quanting):
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
Detailed Reddit post with build commands, test-backend-ops, multi-GPU scheduler flags (GGML_SCHED_MAX_SPLIT_INPUTS=128), and A100 mentions:
https://www.reddit.com/r/LocalLLM/comments/1taclsw/deepseek_v4_in_llamacpp_flash_pro_cuda_metal/
Upstream llama.cpp tracking issue for V4:
https://github.com/ggml-org/llama.cpp/issues/22319

2. Patched vLLM (Genesis-style patches + Ampere fork for MTP, PagedAttention, FP8 KV with BF16 fallbacks)Lasimeri Ampere-specific fork (pyref kernel replacements + SM86/80 support for V4-Flash):
https://github.com/Lasimeri/vllm-dsv4-ampere
Official vLLM blog announcing Day-0 DeepSeek-V4 (Flash/Pro) support (MTP, hybrid attention, long-context recipes):
https://vllm.ai/blog/2026-04-24-deepseek-v4
Additional vLLM forks/PRs referenced in community (e.g., jasl/vllm for multi-node recipes):
https://github.com/jasl/vllm (pin specific commits for stability, as noted in NVIDIA forums)

3. TensorRT-LLM + Triton Inference Server (Native Ampere support, EP/TP/PP, production serving)Main TensorRT-LLM GitHub issue for DeepSeek-V4 support (includes PyTorch backend PRs and attention kernels):
https://github.com/NVIDIA/TensorRT-LLM/issues/13431
and follow-up: https://github.com/NVIDIA/TensorRT-LLM/issues/13435
TensorRT-LLM releases page (search for DeepSeek V4 / Flash entries — multiple PRs for FP4/FP8 kernels, routing, and MoE improvements):
https://github.com/NVIDIA/TensorRT-LLM/releases
KV-cache FP8 layout discussion (relevant for hybrid attention on Ampere):
https://github.com/NVIDIA/TensorRT-LLM/issues/14327

4. ik_llama.cpp (Advanced GGUF with SOTA quants, hybrid CPU/GPU, MLA/FlashMLA for DeepSeek-style models)ikawrakow fork (optimized for better CPU/hybrid performance, custom quants, and DeepSeek models):
https://github.com/ikawrakow/ik_llama.cpp
(Docs: https://ikawrakow-ik_llama-cpp.mintlify.app/)

5. MTP Speculative Decoding + Related TechniquesThese are integrated into the engines above (native single-layer MTP head in V4-Flash).  vLLM MTP docs are in the official blog linked above.  
llama.cpp MTP support is in the cchuter fork (check README and recent commits).  
For expert budgeting / MoE-Spec extensions, watch vLLM and SGLang repos (not fully detailed in club-3090 yet, but the repo’s MTP preservation philosophy applies directly).

Additional useful links:Official DeepSeek-V4-Flash model card (architecture details, FP4+FP8 mixed weights, 1M context):
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
NVIDIA forum threads with multi-GPU / Ampere V4-Flash recipes (TP/EP examples):
https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309

These links point to the exact repos, branches, and discussions where the patches, kernels, and configs live. Start with the cchuter llama.cpp fork + teamblobfish GGUF for your 4x A100 (it already includes the multi-GPU scheduler fix and sm_80 FP8 emulation path). The club-3090 repo’s patching philosophy (cudagraph fixes, MTP preservation, cliff avoidance) can be ported/cherry-picked onto the vLLM or TensorRT-LLM paths.If you pick one approach and want me to pull specific files, commands, or diff the patches against club-3090, just say which one!
