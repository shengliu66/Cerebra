<p align="center">
  <img src="figure/Cerebra_logo.png" alt="Cerebra Logo" width="460" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img alt="LLM" src="https://img.shields.io/badge/LLM-Agent%20Orchestration-7B61FF" />
  <a href="https://arxiv.org/pdf/2603.21597">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Read%20Now-EA4335" />
  </a>
  <a href="https://cerebra-health.com">
    <img alt="Website" src="https://img.shields.io/badge/Website-Visit-0A66C2" />
  </a>
  <a href="https://app.cerebra-health.com">
    <img alt="Demo" src="https://img.shields.io/badge/Demo-Try%20It-16A34A" />
  </a>
</p>

## 🧠 Overview

Cerebra is a multidisciplinary AI board that works along with physicians to make clinical decisions. 

## ⚙️ Installation

Create and activate a fresh environment:

```sh
conda create -n cerebra python=3.10
conda activate cerebra
```

Install dependencies and package:

```sh
pip install -r requirements.txt
pip install -e .
```

## 🔐 Environment Configuration

Create a `.env` file under `cerebra/agents/` and configure API keys.
Currently, OpenAI is the main supported setup (for `gpt4o`), but other key placeholders are kept for future engine support.

Example:

```sh
# cerebra/agents/.env

# Used for LLM-powered modules and tools
OPENAI_API_KEY=<your-api-key-here>    # OpenAI
ANTHROPIC_API_KEY=<your-api-key-here> # Anthropic
TOGETHER_API_KEY=<your-api-key-here>  # TogetherAI
DEEPSEEK_API_KEY=<your-api-key-here>  # DeepSeek
GOOGLE_API_KEY=<your-api-key-here>    # Gemini
XAI_API_KEY=<your-api-key-here>       # Grok

# EHR feature header files (used for evidence / feature-name mapping)
# EHR header files contain feature names in format: "Code: Description"
# Example format:
# Phecode_ID_015.2: Clostridium difficile
# Phecode_ID_089.1: Bacterial infections
# Phecode_ID_069: Other specified viral infections
CEREBRA_EHR_HEADER_PATH=/path/to/ehr_headers.txt

# MRI data path (used for image agent visualization)
# Directory containing MRI scan files in .mgz format
# Files should be named as: {subject_id}_{session_id}_mri.mgz
MRI_BASE_PATH=/path/to/MRI_scans/
```

## 🗂️ Data Agent

`DataAgent` (`cerebra/agents/data_agent.py`) is the entry point for all data access. It supports four modes: `local` (offline `.pkl`/`.csv` files), `training` (PostgreSQL cohort retrieval + train/val/test split), `inference` (single-patient history, no labels), and `exploration` (ad hoc schema/cohort/history queries).

See [`cerebra/tools/data_agent/README.md`](cerebra/tools/data_agent/README.md) for full usage, data formats, and SQL mode details.

## 🚀 Running the system

Use `tasks/run_cerebra.py` as the main entry point.

### ▶️ Basic run

```sh
python tasks/run_cerebra.py --patient_id 0 --year 3 --institution NYU
```

### 🛠️ Common options

- `--llm_engine` (default: `gpt-4o`)
- `--output_json <path>` to save full output
- `--diagnosis True|False`
- `--time_to_event True|False`
- `--volume True|False` (for image volume CSV mode)

### 📝 Example with output JSON

```sh
python tasks/run_cerebra.py \
  --patient_id 0 \
  --year 3 \
  --institution NYU \
  --llm_engine gpt-4o \
  --output_json cerebra_cache/run_patient_0.json
```

### Supported LLM Engines

We support a broad range of LLM engines, including GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and more.

| Model Family | Engines (Multi-modal) | Engines (Text-Only) | Official Model List |
|--------------|-------------------|--------------------| -------------------- |
| OpenAI | `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`,  `gpt-4.1`,  `gpt-4.1-mini`, `gpt-4.1-nano`, `o1`, `o3`, `o1-pro`, `o4-mini` | `gpt-3.5-turbo`, `gpt-4`, `o1-mini`, `o3-mini` | [OpenAI Models](https://platform.openai.com/docs/models) |
| Azure OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o1`, `o3`, `o1-pro`, `o4-mini` | `gpt-3.5-turbo`, `gpt-4`, `o1-mini`, `o3-mini` | [Azure OpenAI Models](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#models) |
| Anthropic | `claude-3-haiku-20240307`, `claude-3-sonnet-20240229`, `claude-3-opus-20240229`, `claude-3-5-sonnet-20240620`, `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-7-sonnet-20250219` | | [Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models/all-models) |
| TogetherAI | Most multi-modal models, including `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `Qwen/QwQ-32B`, `Qwen/Qwen2-VL-72B-Instruct` | Most text-only models, including `meta-llama/Llama-3-70b-chat-hf`, `Qwen/Qwen2-72B-Instruct` | [TogetherAI Models](https://api.together.ai/models) |
| DeepSeek |  | `deepseek-chat`, `deepseek-reasoner` | [DeepSeek Models](https://api-docs.deepseek.com/quick_start/pricing) |
| Gemini | `gemini-1.5-pro`, `gemini-1.5-flash-8b`, `gemini-1.5-flash`, `gemini-2.0-flash-lite`, `gemini-2.0-flash`, `gemini-2.5-pro-preview-03-25` |  |  [Gemini Models](https://ai.google.dev/gemini-api/docs/models) |
| Grok | `grok-2-vision-1212`, `grok-2-vision`, `grok-2-vision-latest` | `grok-3-mini-fast-beta`, `grok-3-mini-fast`, `grok-3-mini-fast-latest`, `grok-3-mini-beta`, `grok-3-mini`, `grok-3-mini-latest`, `grok-3-fast-beta`, `grok-3-fast`, `grok-3-fast-latest`, `grok-3-beta`, `grok-3`, `grok-3-latest` | [Grok Models](https://docs.x.ai/docs/models#models-and-pricing) |
| vLLM | Various vLLM-supported models, for example, `Qwen2.5-VL-3B-Instruct` and `Qwen2.5-VL-72B-Instruct`. You can also use local checkpoint models for customization and local inference. ([Example-1](https://github.com/octotools/octotools/blob/main/examples/notebooks/baseball_query_local_model_qwen.ipynb), [Example-2](https://github.com/octotools/octotools/blob/main/examples/notebooks/baseball_query_parallel_inference.ipynb))| Various vLLM-supported models, for example, `Qwen2.5-1.5B-Instruct`. You can also use local checkpoint models for customization and local inference. | [vLLM Models](https://docs.vllm.ai/en/latest/models/supported_models.html) |
| LiteLLM | Any model supported by LiteLLM, including models from OpenAI, Anthropic, Google, Mistral, Cohere, and more. | Any model supported by LiteLLM, including models from OpenAI, Anthropic, Gemini, Mistral, Cohere, and more. | [LiteLLM Models](https://docs.litellm.ai/docs/providers) |
| Forge | Any Forge-supported models via `forge/Provider/model-name` (e.g., `forge/OpenAI/gpt-4o-mini`). | Same as multi-modal column. | [Forge Models](https://forge.tensorblock.co) |
| Ollama | Any model supported by Ollama, such as `DeepSeek-R1`, `Qwen 3`, `Llama 3.3`, `Gemma 3`, and other models. | Any model supported by Ollama, such as `Qwen 2.5‑VL`. | [Ollama Models](https://ollama.ai/library) |


> Note: If you are using TogetherAI models, please ensure have the prefix 'together-' in the model string, for example, `together-meta-llama/Llama-4-Scout-17B-16E-Instruct`. For VLLM models, use the prefix 'vllm-', for example, `vllm-meta-llama/Llama-4-Scout-17B-16E-Instruct`. For LiteLLM, use the prefix 'litellm-', for example, `litellm-gpt-4o` or `litellm-claude-3-sonnet-20240229`. For Ollama, use the prefix 'ollama-', for example, `ollama-qwen3:latest`. For Forge, use the prefix 'forge/', for example, `forge/OpenAI/gpt-4o-mini`, and set `FORGE_API_KEY` (optional `FORGE_API_BASE`). For other custom engines, you can edit the [factory.py](https://github.com/OctoTools/OctoTools/blob/main/octotools/engine/factory.py) file and add its interface file to add support for your engine. Your pull request will be warmly welcomed!


## Resources

### Inspiration

This project draws inspiration from several remarkable projects:

- 📘 [TextGrad](https://github.com/mert-y/textgrad) – We admire and appreciate TextGrad for its innovative and elegant framework design.
- 📗 [OctoTools](https://github.com/octotools/octotools) – A open-sourced agentic framework for tool usage.


### Citation
```bibtex
}
```

### Contributors

We are truly looking forward to open-source contributions to Cerebra! If you are interested in contributing, collaborating, or reporting issues, don't hesitate to contact us at [shengliu888@gmail.com](mailto:hengliu888@gmail.com). 

We are also looking forward to your feedback and suggestions!

