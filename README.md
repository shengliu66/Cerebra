<p align="center">
  <img src="figure/Cerebra_logo.png" alt="Cerebra Logo" width="460" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img alt="LLM" src="https://img.shields.io/badge/LLM-Agent%20Orchestration-7B61FF" />
  <a href="https://example.com/paper">
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

Cerebra is an AI Copilot that works along with physicians to make decisions. 

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
