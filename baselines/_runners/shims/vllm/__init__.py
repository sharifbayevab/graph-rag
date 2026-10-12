"""Import shim: HippoRAG 2.0.0a4 / PropRAG import vllm at module level for their offline-OpenIE path, which these runs
never use (openie_mode="online"). vllm 0.6.6 has no aarch64 / CUDA 13 build for the Spark GB10."""


class _Unavailable:
    def __init__(self, *a, **kw):
        raise RuntimeError("vllm shim: offline vLLM mode is not available in this environment")


SamplingParams = LLM = _Unavailable
