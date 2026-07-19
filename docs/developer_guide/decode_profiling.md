# Decode profiling

The fine-grained decode ranges are disabled by default. Enable them before the
server process starts:

```bash
SGLANG_DECODE_PROFILE=1 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=/path/to/decode-profile \
  python -m sglang.launch_server \
  ... \
  --disable-cuda-graph
```

Use the profiling HTTP endpoint or the serving benchmark's
`--profile-activities CUDA_PROFILER` option to delimit the Nsight Systems
capture after warmup. `--disable-cuda-graph` is required to expose ranges
inside each transformer layer. With CUDA Graph enabled, only the
`sglang.decode.model_runner.graph_replay` host range is visible at runtime.

The marker hierarchy includes:

- `sglang.decode.model_runner.*`: metadata, eager model forward, graph replay
- `sglang.decode.layer.NN.*`: communication, attention prepare/core, MoE
- `sglang.decode.kt.layer.NN.*`: CPU submit, mask/remap, GPU experts, CPU sync,
  and merge
- `sglang.decode.logits` and `sglang.decode.sampling.*`: output processing

The instrumentation does not call `torch.cuda.synchronize()`. Use
`SGLANG_KT_HYBRID_TIMING_DEEP=1` only for one-shot diagnosis when deliberately
serializing the KT CPU/GPU overlap is acceptable.
