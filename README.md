<p align="center">
  <img src="figure/Cerebra_logo.png" alt="Cerebra Logo" width="460" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img alt="LLM" src="https://img.shields.io/badge/LLM-Agent%20Orchestration-7B61FF" />
  <img alt="Data" src="https://img.shields.io/badge/Data-Pickle%20%26%20CSV-2E8B57" />
  <a href="https://example.com/paper">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Read%20Now-EA4335" />
  </a>
  <a href="https://cerebra-health.com">
    <img alt="Website" src="https://img.shields.io/badge/Website-Visit-0A66C2" />
  </a>
  <a href="https://example.com/demo">
    <img alt="Demo" src="https://img.shields.io/badge/Demo-Try%20It-16A34A" />
  </a>
</p>

## 🧠 Overview

Cerebra is an LLM-powered framework for clinical/evaluation workflows.

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
# Example: /gpfs/data/razavianlab/mri_scans/
MRI_BASE_PATH=/path/to/MRI_scans/
```

## 🗂️ Data (Database) Setup

This project uses file-based clinical datasets (pickle/CSV), not a SQL database.
The data loading logic is implemented in `cerebra/agents/data_agent.py`.


### 📍 dataset format

- NYU rolling data root: `/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/20251207/rolling_not_long_island/`
- LongIsland rolling data root: `/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/20251207/rolling_long_island/`
- Diagnosis data:
  - `/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/diagnosis_not_long_island/`
  - `/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/diagnosis_long_island/`
- Time-to-event data:
  - `/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/mri_indexed_not_long_island/time_to_event/`
  - `/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/mri_indexed_long_island/time_to_event/`

### 📦 Required files (by task mode)

For standard prediction (`diagnosis=False`, `time_to_event=False`), data is loaded from:
`<root>/180days_blackout_<year>yr_label/`

Common label files:

- `Y_train.pkl`
- `Y_val.pkl`
- `Y_test.pkl`

Agent-specific feature files:

- EHR: `X_ehr_train_4447_header_phecode.pkl`, `X_ehr_val_4447_header_phecode.pkl`, `X_ehr_test_4447_header_phecode.pkl`
- Notes: `X_note_train.pkl`, `X_note_val.pkl`, `X_note_test.pkl`
- MRI path mode: `X_mri_train.pkl`, `X_mri_val.pkl`, `X_mri_test.pkl`
- MRI volume mode: `X_mri_volume_train.csv`, `X_mri_volume_val.csv`, `X_mri_volume_test.csv`

If `time_to_event=True`, additional label files are required:

- `Y_time_to_event_train.pkl`
- `Y_time_to_event_val.pkl`
- `Y_time_to_event_test.pkl`

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
