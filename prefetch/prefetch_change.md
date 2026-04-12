# HiSparse Stage B Prefetch — Change Summary

## Overview

Adds a one-layer-ahead prefetch mechanism to HiSparse sparse attention.
During layer N's forward, probes layer N+1's indexer to predict which KV
entries will be needed, then prefetches missing entries from host to device
on a separate CUDA stream — overlapping H2D copies with compute.

Layer 0 is excluded (no predecessor to probe from). The existing
non-prefetch path is preserved as the default.

## Configuration

Enabled via `hisparse_config` JSON or the launch script knobs:

```bash
ENABLE_PREFETCH=1 bash prefetch/launch_hisparse_ds_v32.sh

# Optional tuning
ENABLE_PREFETCH=1 PREFETCH_TOPK=1024 NUM_MAX_PREFETCH=256 bash prefetch/launch_hisparse_ds_v32.sh
```

| Knob | Default | Meaning |
|------|---------|---------|
| `enable_prefetch` | `false` | Master switch |
| `prefetch_topk` | `top_k` (2048) | Probe top-k size. Supports values < `index_topk` (truncates after fused topk kernel). |
| `num_max_prefetch` | `top_k` (2048) | Max H2D copies per request in the prefetch kernel |

## Files Changed

### Config / wiring

| File | Change |
|------|--------|
| `srt/mem_cache/sparsity/core/sparse_coordinator.py` | Added `enable_prefetch`, `prefetch_topk`, `num_max_prefetch` to `SparseConfig` |
| `srt/mem_cache/sparsity/factory.py` | Parse + validate new knobs from `hisparse_config` JSON |
| `srt/model_executor/model_runner.py` | Pass new params to `HiSparseCoordinator()` |

### JIT CUDA kernel

| File | Change |
|------|--------|
| `jit_kernel/csrc/hisparse.cuh` | Added `HiSparseMode` enum (FULL/PREFETCH/RESOLVE), `Mode` template param, `num_max_prefetch` runtime arg, `stats_out` per-request hit/miss export. PREFETCH mode: caps miss copies, skips `top_k_device_locs` writes. |
| `jit_kernel/hisparse.py` | Exposed mode constants, added `mode`, `num_max_prefetch`, `stats_out` params |

### Coordinator

| File | Change |
|------|--------|
| `srt/managers/hisparse_coordinator.py` | `prefetch_stream`, `prefetch_selected_pages()`, `_wait_prefetch_if_needed()`. `swap_in_selected_pages()` waits on prefetch then runs RESOLVE mode. Per-layer stats accumulation + periodic logging. |

### Indexer

| File | Change |
|------|--------|
| `srt/layers/attention/nsa/nsa_indexer.py` | `_get_q_only_bf16()` (query-only, no key). `forward_probe()` — lightweight path skipping K-store and all K computation. `_get_topk_paged()` accepts optional `topk` override. Handles `prefetch_topk < index_topk` via truncation. |

### Model layer loop

| File | Change |
|------|--------|
| `srt/models/deepseek_v2.py` | `compute_probe_input()`, `compute_probe_q_lora()`, `launch_prefetch()` on `DeepseekV2DecoderLayer`. Layer loop calls `launch_prefetch()` for next layer after each layer's forward. |

### Launch script

| File | Change |
|------|--------|
| `prefetch/launch_hisparse_ds_v32.sh` | Knob-based config: `ENABLE_PREFETCH`, `PREFETCH_TOPK`, `NUM_MAX_PREFETCH` env vars |
| `prefetch/req.sh` | Added `prefill_tokens` / `decode_tokens` to output, simplified consistency check |

## Control Flow

```
Layer N forward completes → (hidden_states, residual, topk_indices)

if enable_prefetch AND decode AND next_layer.use_nsa:
  [main stream] probe_x = input_layernorm_{N+1}(hidden_states + residual)
  [main stream] q_lora  = q_a_layernorm_{N+1}(q_a_proj_{N+1}(probe_x))
  prefetch_stream.wait_stream(main_stream)
  [prefetch stream] probe_topk = indexer.forward_probe(...)   # query-only, no K-store
  [prefetch stream] PREFETCH kernel(probe_topk, layer=N+1)    # caps at num_max_prefetch

Layer N+1 forward:
  real indexer runs (with K-store) on main stream
  swap_in_selected_pages(real_topk, layer=N+1):
    main_stream.wait_stream(prefetch_stream)  # ensure prefetch done
    RESOLVE kernel(real_topk)                 # prefetched tokens are hits
    → top_k_device_locs for attention
```

## Stats Logging

The kernel exports per-request `[hits, misses]` via `stats_out`. The coordinator
accumulates across requests and logs every 40 decode steps:

```
HiSparse resolve miss rate (40 steps): L0:4.5%(3654/81920) L1:1.5%(1200/81920) ...
HiSparse prefetch hit rate (40 steps): L0:0.0%(0/0) L1:94.1%(77078/81920) ...
```

## Verified Results (4-layer truncated DS-V3.2, H100)

### Resolve miss rate

| Layer | No Prefetch | prefetch_topk=2048 | prefetch_topk=1024 |
|-------|------------|-------------------|-------------------|
| L0 | ~4.5% | ~4.5% (no prefetch) | ~4.5% (no prefetch) |
| L1 | ~6.0% | ~1.5% | ~4.6% |
| L2 | ~6.5% | ~5.5% | ~6.0% |
| L3 | ~4.5% | ~1.5% | ~4.3% |

### CudaGraph

All configurations captured and replayed successfully (`cuda graph: True` in decode logs).

## Known Limitations

1. Layer 0 gets no prefetch (no predecessor)
2. Probe K-cache for the next layer is missing the current decode token's K (noise on position seq_len-1, harmless)
3. Wrong-evict is ignored: prefetch may evict slots the real top-k needs
4. `prefetch_topk < index_topk` works but computes full `index_topk` then truncates (fused topk kernel constraint)
5. Probe input + q_lora computed on main stream (not fully overlapped)
6. Only decode path; prefill, speculative, MTP, PD-direct unaffected
