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

## 🗂️ Data Agent

`DataAgent` (`cerebra/agents/data_agent.py`) is the entry point for all data access. It supports four operational modes selected via the `mode` argument.

### Modes

| Mode | Data source | Use case |
|---|---|---|
| `local` | Pre-processed `.pkl` / `.csv` files on disk | Offline experiments, reproducible runs |
| `training` | PostgreSQL via natural-language query | Retrieve cohort + labels, split into train/val/test |
| `inference` | PostgreSQL via natural-language query | Retrieve a single patient's longitudinal history (no labels) |
| `exploration` | PostgreSQL via natural-language query | Ad hoc: schema inspection, cohort discovery, patient history, similar-patient search |

---

### `local` mode

Pass a `file_paths` dict mapping split/role names to file paths. Files can be `.pkl` (pickled Python objects) or `.csv`.

```python
from cerebra.agents.data_agent import DataAgent

agent = DataAgent()
metadata = agent.run(
    mode="local",
    file_paths={
        "train_data":         "path/to/X_train.pkl",
        "train_labels":       "path/to/Y_train.pkl",
        "validation_data":    "path/to/X_val.pkl",
        "validation_labels":  "path/to/Y_val.pkl",
        "test_data":          "path/to/X_test.pkl",
        "test_labels":        "path/to/Y_test.pkl",
        "demographics":       "path/to/demographics.pkl",  # optional
    },
    agent_name="ehr_agent",   # "ehr_agent" | "note_agent" | "image_agent"
    patient_id=0,             # optional: slice test split to one patient
)
```

#### Required data formats per agent

**`ehr_agent`**
- `*_data`: `list[scipy.sparse.csr_matrix]` — one sparse matrix per patient, shape `(timesteps, n_features)`
- `*_labels`: `list[int]` — binary labels (0 / 1), one per patient

**`note_agent`**
- `*_data`: `list[list[str]]` — outer list is patients; inner list is that patient's clinical notes as strings
- `*_labels`: `list[int]`

**`image_agent`**
- `*_data`: `list[str]` — file paths to pre-processed MRI files, one per patient
- `*_labels`: `list[int]`

**`demographics`** (optional, used when `patient_id` is set): `list[dict]`, one dict per patient, containing `age` (float), `gender` (str), and lab feature keys for APOE and memory status (int, value > 0 means positive).

---

### SQL modes (`training` / `inference` / `exploration`)

These modes require a running PostgreSQL database and an OpenAI-compatible LLM for natural-language-to-SQL translation.

Each call targets a single modality. Run once per modality needed.

```python
# Training: retrieve cohort + labels and split into train/val/test (one modality per call)
agent.run(
    mode="training",
    postgresql_database_url="postgresql+psycopg2://user:pass@host:5432/db",
    natural_language_query=(
        "I want to analyze dementia risk over the next 2 years, "
        "from EHR Records, Clinical Notes, and MRI Imaging."
    ),
    modality="ehr",   # one of: "ehr" | "notes" | "images"
    train_ratio=0.70,
    val_ratio=0.15,
    balance=True,
)

# Inference: single patient longitudinal history (no outcome labels)
agent.run(
    mode="inference",
    postgresql_database_url="postgresql+psycopg2://user:pass@host:5432/db",
    patient_id="0",
    natural_language_query=(
        "Retrieve the full medical history for Patient MRN_12345 to assess "
        "dementia risk, including EHR records, clinical notes, and MRI imaging."
    ),
    modality="notes",   # one of: "ehr" | "notes" | "images"
)

# Exploration: ad hoc database interrogation
agent.run(
    mode="exploration",
    postgresql_database_url="postgresql+psycopg2://user:pass@host:5432/db",
    natural_language_query="Give me diabetes-related EHR records for patient MRN_12345.",
    exploration_type="patient_history",  # "schema" | "cohort" | "patient_history" | "similar_patients"
    patient_id="MRN_12345",
)
```

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
