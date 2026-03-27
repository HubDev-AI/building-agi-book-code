# AGI MVP -- Companion Code

Companion code for **"Building AGI: A Hands-On Guide to Artificial General Intelligence"**, available on [Amazon KDP](https://www.amazon.com).

## Quick Start

```bash
git clone https://github.com/HubDev-AI/building-agi-book-code.git
cd building-agi-book-code

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Getting API Keys

The code uses any OpenAI-compatible API. The book defaults to DeepSeek.

### DeepSeek (default)

1. Sign up at [platform.deepseek.com](https://platform.deepseek.com)
2. Create an API key at [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
3. Top up your balance (new accounts get free credits)

```bash
export DEEPSEEK_API_KEY="sk-..."
```

The code uses two DeepSeek models:
- **deepseek-chat** (DeepSeek V3) -- fast model for quick tasks
- **deepseek-reasoner** (DeepSeek R1) -- reasoning model for complex problems

### Qwen (alternative)

1. Create an account at [alibabacloud.com](https://www.alibabacloud.com)
2. Activate **Model Studio** and create an API key at the [Key Management page](https://bailian.console.alibabacloud.com/#/api-key)

```bash
export DASHSCOPE_API_KEY="sk-..."
```

Then configure in code:

```python
from agi_mvp.foundation import ModelConfig

config = ModelConfig(
    fast_model="qwen3-32b",
    fast_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    reasoning_model="deepseek-reasoner",
    reasoning_base_url="https://api.deepseek.com",
)
```

## Running

```bash
export DEEPSEEK_API_KEY="sk-..."
python -m agi_mvp.main
```

Type a question or task and press Enter. Type `quit` to exit.

## Local Models (Optional)

For running models locally with vLLM instead of using APIs:

```bash
pip install vllm

# Fast model (Qwen3-32B on a single GPU)
vllm serve Qwen/Qwen3-32B --port 8001

# Reasoning model (DeepSeek-R1 needs 2+ GPUs)
vllm serve deepseek-ai/DeepSeek-R1-0528-AWQ \
  --tensor-parallel-size 2 --port 8000
```

Then point the config at localhost:

```python
config = ModelConfig(
    fast_model="Qwen/Qwen3-32B",
    fast_base_url="http://localhost:8001/v1",
    reasoning_model="deepseek-ai/DeepSeek-R1-0528-AWQ",
    reasoning_base_url="http://localhost:8000/v1",
)
```

## Episodic Memory

Episodic memory uses Qdrant for vector search. For development, use in-memory mode (no setup needed). For persistence:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## License

This code accompanies the book. See the book for full context and explanations.
