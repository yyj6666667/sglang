# MiniMax-M3 Support

SGLang supports [MiniMax-M3](https://huggingface.co/MiniMaxAI) with native MXFP8 inference on Hopper-class GPUs (H100/H20) and optional KT-Kernel offload for CPU experts (large hybrid deploys). M3 is a 230B sparse model with 60 MoE layers, 128 routed experts + 1 shared expert, sigmoid routing, and a native MXFP8 weight format.



## Installation

```bash
git clone https://github.com/yyj6666667/sglang.git
cd sglang
git checkout feat/minimax-m3
pip install -e "python[all]"

# Optional: KT-Kernel CPU offload (only needed for hybrid deploys)
pip install kt-kernel
```

## Launch

### Pure GPU (Recommended — full speed)

8×H20 / H100, all experts on GPU.

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/Minimax-M3-preview \
    --tp 8 \
    --quantization mxfp8 \
    --moe-runner-backend triton \
    --mem-fraction-static 0.70 \
    --chunked-prefill-size 4096 \
    --trust-remote-code \
    --port 30000
```

### Hybrid (CPU expert offload)

2× / 4× H20 / H100, most experts on CPU. Picks N hottest experts per rank to keep on GPU; rest dispatch to CPU via KT-Kernel.

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/Minimax-M3-preview \
    --tp 2 \
    --quantization mxfp8 \
    --moe-runner-backend triton \
    --kt-weight-path /path/to/Minimax-M3-preview \
    --kt-method MXFP8 \
    --kt-cpuinfer 64 \
    --kt-threadpool-count 2 \
    --kt-num-gpu-experts 8 \
    --kt-gpu-prefill-token-threshold 500 \
    --mem-fraction-static 0.30 \
    --chunked-prefill-size 4096 \
    --trust-remote-code \
    --port 30000
```

Tuning notes:

- `--kt-num-gpu-experts N`: experts kept on GPU per TP rank. Larger N → faster but more GPU memory. Typical 8–40.
- `--kt-cpuinfer`: CPU worker threads for offloaded experts. Set to physical core count.
- `--kt-threadpool-count`: NUMA pools; usually equals socket count.
- `--kt-gpu-prefill-token-threshold T`: chunks of ≥ T tokens trigger a full-GPU prefill fallback (faster long-prompt prefill).

### Function Calling / Reasoning

Add these flags to either launch above to enable tool-call parsing and `<mm:think>` reasoning extraction:

```bash
--tool-call-parser minimax-m3 \
--reasoning-parser minimax-m3
```

## Usage

### Chat completion

```python
import openai
client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.0,
    max_tokens=64,
)
print(response.choices[0].message.content)
# Paris.
```

### Thinking mode

M3 supports request-level thinking control via `chat_template_kwargs.thinking_mode`. Three values are accepted; reasoning content (if any) is returned under `message.reasoning_content`.

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_body={"chat_template_kwargs": {"thinking_mode": "disabled"}},   # or "enabled" / "adaptive"
    max_tokens=50,
)
print("content:",            response.choices[0].message.content)            # "4"
print("reasoning_content:",  response.choices[0].message.reasoning_content)  # None
```

| `thinking_mode` | Behavior |
|---|---|
| `"enabled"` | Force chain-of-thought; `<mm:think>` start tag is prefilled by the template. |
| `"disabled"` | Suppress thinking; closing tag prefilled. |
| `"adaptive"` (default) | Model self-decides; detector handles emitted `<mm:think>` blocks. |

### Tool calls

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Shanghai?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=200,
)
print(response.choices[0].message.tool_calls)
# [ChatCompletionMessageToolCall(
#    function=Function(name='get_weather', arguments='{"city": "Shanghai"}'),
#    type='function', ...)]
```

The model emits tool calls in the native `<minimax:tool_call>` XML format; the parser converts to OpenAI `tool_calls` automatically. Structural-tag (xgrammar) constraints are not yet supported for M3 because of the XML grammar.

## Benchmarks

Measured on 8×NVIDIA H20 with the configurations above. GSM8K (5-shot, temperature=0, 200 questions).

| Config | TP | GPU experts | Accuracy | Decode tok/s (bs=1) |
|---|---|---|---|---|
| Pure GPU | 8 | 128 (all) | 87 % | ~5 |
| Hybrid | 2 | 8 (CPU offload) | 89 % | ~19 |

Throughput dominated by 8-way TP all-reduce in the pure-GPU case (one all-reduce per attention + one per MoE down per layer × 60 layers × 2 = 120 small-message all-reduces per decode step). Scaling decode batch (`--max-running-requests`) amortizes this and gives much higher per-server tok/s.

## Known Limitations

- xgrammar / structural-tag constraints not yet supported for M3 tool calls (model uses XML grammar).
- KT-Kernel hybrid mode is single-node only; for multi-node hybrid, use the pure-GPU path with `--ep-size > 1` instead.
- Cuda graph capture: `--cuda-graph-max-bs 1` is recommended for decode-only profiling; for serving raise it to your max batch size.
