# Data Agent

`DataAgent` (`cerebra/agents/data_agent.py`) is the unified data-access layer for Cerebra. Select a mode via the `mode` argument to `agent.run()`.

## Modes

| Mode | Data source | Typical use |
|---|---|---|
| `local` | Pre-processed `.pkl` / `.csv` files on disk | Offline experiments, reproducible runs, testing |
| `training` | PostgreSQL + NL-to-SQL | Retrieve cohort + labels, split into train/val/test |
| `inference` | PostgreSQL + NL-to-SQL | Retrieve a single patient's history (no labels) |
| `exploration` | PostgreSQL + NL-to-SQL | Ad hoc schema inspection, cohort discovery, similar-patient search |

---

## `local` mode

No database required. Pass a `file_paths` dict and the agent loads everything from disk.

```python
from cerebra.agents.data_agent import DataAgent

agent = DataAgent()
metadata = agent.run(
    mode="local",
    file_paths={
        "train_data":        "path/to/X_train.pkl",
        "train_labels":      "path/to/Y_train.pkl",
        "validation_data":   "path/to/X_val.pkl",
        "validation_labels": "path/to/Y_val.pkl",
        "test_data":         "path/to/X_test.pkl",
        "test_labels":       "path/to/Y_test.pkl",
        "demographics":      "path/to/demographics_test.pkl",  # optional
        "trained_model":     "path/to/model.joblib",           # optional
    },
    agent_name="ehr_agent",   # "ehr_agent" | "note_agent" | "image_agent"
    patient_id=0,             # optional — slices test split to one patient (int index)
)
```

### Data formats per agent

**`ehr_agent`**
- `*_data`: `list[scipy.sparse.csr_matrix]` — one matrix per patient, shape `(timesteps, n_features)`
- `*_labels`: `list[int]` — binary labels (0 / 1)
- `demographics`: `list[dict]` — one dict per patient; required keys: `age` (float), `gender` (str), APOE and memory-complaint keys (int, >0 means positive)
- `trained_model`: path to a `.joblib` file — populates `model['trained_model']` in the returned metadata so the EHR inference tool can load it

**`note_agent`**
- `*_data`: `list[list[str]]` — outer list is patients; inner list is that patient's clinical notes
- `*_labels`: `list[int]`
- `trained_model`: path to a `.joblib` file (optional)

**`image_agent`**
- `*_data`: `list[str]` (file paths to pre-processed MRI files) **or** a `.csv` file (MRI volume measurements)
- `*_labels`: `list[int]`
- `trained_model`: path to a `.joblib` file (optional)

> **Note:** Labels are shared across modalities — the same `Y_train.pkl` / `Y_val.pkl` / `Y_test.pkl` can be reused for all three agents.

### Environment variable

`CEREBRA_EHR_HEADER_PATH` must point to a plain-text file with one feature name per line. Required when the EHR inference tool computes feature-importance evidence.

---

## `training` mode

Queries a PostgreSQL database for a cohort matching the natural-language description, then splits patients into train/val/test sets.

```python
agent.run(
    mode="training",
    postgresql_database_url="postgresql+psycopg2://user:pass@host:5432/db",
    natural_language_query=(
        "Retrieve dementia patients and matched controls with at least 2 years "
        "of follow-up from EHR Records, Clinical Notes, and MRI Imaging."
    ),
    modality="ehr",       # "ehr" | "notes" | "images"
    train_ratio=0.70,     # default
    val_ratio=0.15,       # remainder goes to test
    balance=True,         # undersample majority class in train split
)
```

**Dependencies:** `sqlalchemy`, `psycopg2-binary`, OpenAI API key for NL-to-SQL.

---

## `inference` mode

Retrieves the longitudinal history of a single patient without outcome labels.

```python
agent.run(
    mode="inference",
    postgresql_database_url="postgresql+psycopg2://user:pass@host:5432/db",
    patient_id="MRN_12345",
    natural_language_constraint="only records from the past 3 years",  # optional
    modality="notes",   # "ehr" | "notes" | "images"
)
```

---

## `exploration` mode

Ad hoc interrogation of the database. Supports four `exploration_type` values:

| `exploration_type` | What it does |
|---|---|
| `schema` | Lists tables, columns, and types |
| `cohort` | Finds patients matching NL criteria + summary statistics |
| `patient_history` | Full timeline for a specific patient |
| `similar_patients` | Patients with similar clinical profiles |

```python
agent.run(
    mode="exploration",
    postgresql_database_url="postgresql+psycopg2://user:pass@host:5432/db",
    natural_language_query="Show me the schema of the EHR tables.",
    exploration_type="schema",
)

agent.run(
    mode="exploration",
    postgresql_database_url="postgresql+psycopg2://user:pass@host:5432/db",
    natural_language_query="Patients with early-onset dementia diagnosed before age 65.",
    exploration_type="cohort",
)

agent.run(
    mode="exploration",
    postgresql_database_url="postgresql+psycopg2://user:pass@host:5432/db",
    natural_language_query="Full history for patient MRN_12345.",
    exploration_type="patient_history",
    patient_id="MRN_12345",
)
```
