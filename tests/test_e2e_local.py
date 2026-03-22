#!/usr/bin/env python3
"""
End-to-end local-mode test for run_cerebra.py.

Creates minimal synthetic data (no database required), then calls the full
pipeline exactly as a production user would.

Prerequisites
─────────────
1. Install the package:
       cd Cerebra && pip install -e .

2. Set your LLM API key, e.g.:
       export OPENAI_API_KEY=sk-...

3. Run from the Cerebra/ directory:
       python tests/test_e2e_local.py

What it tests
─────────────
• Stage 1 — DataAgent.run(mode="local") for each modality (no LLM).
  Verifies data loading, patient-filtering, and Metadata construction.

• Stage 2 — Full pipeline via tasks/run_cerebra.py (needs LLM).
  Calls SuperAgent → EHR/Note/Image agents → SummaryAgent.
  Writes a JSON result file and prints the final orchestration output.

Data formats created
────────────────────
  ehr_agent   — list of scipy.sparse.csr_matrix, one per patient
  note_agent  — list of list[str], one list of notes per patient
  image_agent — list of file paths (tiny placeholder PNGs)
  labels      — list of int (0/1)
  demographics — list of dict (age, gender, APOE_e4, Memory_complaints)
"""

import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

# Make the Cerebra package importable when running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse

# ══════════════════════════════════════════════════════════════════════════════
# ①  FILL IN YOUR DATA PATHS HERE
#     • Each value is an absolute (or relative) path to a .pkl file.
#     • Set a value to None to auto-generate synthetic data for that split.
# ══════════════════════════════════════════════════════════════════════════════

data_dir = "/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/20251207/rolling_not_long_island/180days_blackout_1yr_label/"

USER_PATHS = {
    # ── EHR: list of scipy.sparse.csr_matrix (one per patient) ───────────────
    "ehr_train_data": os.path.join(data_dir, "X_ehr_train_4447_header_phecode.pkl"),
    "ehr_val_data":   os.path.join(data_dir, "X_ehr_val_4447_header_phecode.pkl"),
    "ehr_test_data":  os.path.join(data_dir, "X_ehr_test_4447_header_phecode.pkl"),

    # ── Note: list of list[str] (one list of notes per patient) ──────────────
    "note_train_data": os.path.join(data_dir, "X_note_train.pkl"),
    "note_val_data":   os.path.join(data_dir, "X_note_val.pkl"),
    "note_test_data":  os.path.join(data_dir, "X_note_test.pkl"),

    # ── Image: list of file paths to MRI files (str) ─────────────────────────
    "image_train_data": os.path.join(data_dir, "X_mri_volume_train.csv"),
    "image_val_data":   os.path.join(data_dir, "X_mri_volume_val.csv"),
    "image_test_data":  os.path.join(data_dir, "X_mri_volume_test.csv"),

    # ── Shared labels (same outcome for all modalities) ───────────────────────
    "train_labels": os.path.join(data_dir, "Y_train.pkl"),
    "val_labels":   os.path.join(data_dir, "Y_val.pkl"),
    "test_labels":  os.path.join(data_dir, "Y_test.pkl"),

    # ── Demographics: list of dict (one per patient per split) ───────────────
    #    Required keys per dict: "age" (float), "gender" (str),
    #                            "APOE_e4" (int 0/1), "Memory_complaints" (int 0/1)
    "demographics_train": os.path.join(data_dir, "demographics_info_train.pkl"),
    "demographics_val":   os.path.join(data_dir, "demographics_info_val.pkl"),
    "demographics_test":  os.path.join(data_dir, "demographics_info_test.pkl"),
}

# ② Other settings ─────────────────────────────────────────────────────────────
PATIENT_IDX = 0    # index into the test split to use as the demo patient
YEAR        = 1    # prediction horizon (years)

# ══════════════════════════════════════════════════════════════════════════════

# Synthetic-data fallback parameters (used only when USER_PATHS entry is None)
N_FEATURES  = 12
N_TRAIN     = 30
N_VAL       = 8
N_TEST      = 5


# ── Data generators ──────────────────────────────────────────────────────────

def _make_ehr_data(n, n_feat, seed):
    """List of sparse row matrices (1 × n_feat), one per patient."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        dense = rng.uniform(0, 10, size=(1, n_feat)).astype(np.float32)
        # ~60 % sparsity
        dense[rng.random((1, n_feat)) < 0.6] = 0.0
        out.append(scipy.sparse.csr_matrix(dense))
    return out


def _make_note_data(n, seed):
    """List of list[str]: one list of clinical-note strings per patient."""
    rng = np.random.default_rng(seed)
    pool = [
        "Patient presents with mild memory impairment. No acute distress.",
        "Radiology: mild white-matter hyperintensities consistent with aging.",
        "Follow-up visit. Reports word-finding difficulty for 6 months.",
        "MRI shows slight cortical atrophy. No acute infarct identified.",
        "Neurological exam within normal limits for age. Gait is steady.",
        "Patient denies headache or focal weakness. Mood is stable.",
    ]
    return [[pool[rng.integers(len(pool))] for _ in range(rng.integers(1, 4))]
            for _ in range(n)]


def _tiny_png_bytes():
    """Return raw bytes of a minimal 1×1 white PNG."""
    def chunk(tag, data):
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xFF\xFF\xFF"))
        + chunk(b"IEND", b"")
    )


def _make_image_data(n, image_dir: Path, seed):
    """List of file paths to placeholder PNG images."""
    rng = np.random.default_rng(seed)
    paths = []
    png = _tiny_png_bytes()
    for i in range(n):
        p = image_dir / f"patient_{i}_mri.png"
        p.write_bytes(png)
        paths.append(str(p))
    rng.shuffle(paths)  # add mild randomness
    return paths


def _make_labels(n, seed):
    """List of binary int labels."""
    rng = np.random.default_rng(seed)
    return list(rng.integers(0, 2, size=n).astype(int))


def _make_demographics(n, seed):
    """List of demographic dicts expected by DataAgent.get_demographic_info()."""
    rng = np.random.default_rng(seed)
    return [
        {
            "age":               float(rng.integers(65, 90)),
            "gender":            "F" if rng.random() > 0.5 else "M",
            "APOE_e4":           int(rng.integers(0, 2)),
            "Memory_complaints": int(rng.integers(0, 2)),
        }
        for _ in range(n)
    ]


def _write_ehr_headers(n_feat: int, path: Path) -> None:
    """Write one feature name per line; used as CEREBRA_EHR_HEADER_PATH."""
    path.write_text("\n".join(f"feature_{i}" for i in range(n_feat)))


# ── Fixture builder ───────────────────────────────────────────────────────────

def _resolve(user_path, fallback_obj, key: str, tmp: Path) -> str:
    """
    If the user supplied a path for *key*, return it as-is.
    Otherwise, pickle *fallback_obj* into tmp/<key>.pkl and return that path.
    """
    if user_path is not None:
        if not Path(user_path).exists():
            raise FileNotFoundError(
                f"USER_PATHS['{key}'] = {user_path!r} does not exist."
            )
        return user_path
    p = tmp / f"{key}.pkl"
    with open(p, "wb") as f:
        pickle.dump(fallback_obj, f)
    return str(p)


def create_test_fixtures(tmp: Path):
    """
    Build the file_paths_per_agent dict consumed by DataAgent and run_cerebra.py.

    For each of the 12 data paths (ehr/note/image/demographics × train/val/test):
      • If you filled in USER_PATHS[key] above → that file is used directly.
      • Otherwise               → synthetic data is generated and saved to tmp/.

    Returns (file_paths_per_agent, headers_file_path).
    """
    (tmp / "ehr").mkdir()
    (tmp / "note").mkdir()
    (tmp / "image").mkdir()

    up = USER_PATHS  # shorthand

    # ── Infer split sizes from user data (for synthetic fallback sizing) ────────
    def _n(path, fallback):
        if path and Path(path).exists():
            if path.endswith(".csv"):
                return len(pd.read_csv(path))
            with open(path, "rb") as f:
                return len(pickle.load(f))
        return fallback

    n_train = _n(up["train_labels"], N_TRAIN)
    n_val   = _n(up["val_labels"],   N_VAL)
    n_test  = _n(up["test_labels"],  N_TEST)

    # ── Shared labels (resolved once, reused by all three agents) ─────────────
    train_labels_path = _resolve(up["train_labels"], _make_labels(n_train, 0), "train_labels", tmp)
    val_labels_path   = _resolve(up["val_labels"],   _make_labels(n_val,   1), "val_labels",   tmp)
    test_labels_path  = _resolve(up["test_labels"],  _make_labels(n_test,  2), "test_labels",  tmp)

    # ── EHR ──────────────────────────────────────────────────────────────────
    headers_file = tmp / "ehr" / "headers.txt"
    _write_ehr_headers(N_FEATURES, headers_file)

    demo_train = _resolve(up["demographics_train"], _make_demographics(n_train, 3), "demographics_train", tmp / "ehr")
    demo_val   = _resolve(up["demographics_val"],   _make_demographics(n_val,   4), "demographics_val",   tmp / "ehr")
    demo_test  = _resolve(up["demographics_test"],  _make_demographics(n_test,  5), "demographics_test",  tmp / "ehr")

    ehr_files = {
        "train_data":        _resolve(up["ehr_train_data"], _make_ehr_data(n_train, N_FEATURES, 0), "ehr_train_data", tmp / "ehr"),
        "validation_data":   _resolve(up["ehr_val_data"],   _make_ehr_data(n_val,   N_FEATURES, 1), "ehr_val_data",   tmp / "ehr"),
        "test_data":         _resolve(up["ehr_test_data"],  _make_ehr_data(n_test,  N_FEATURES, 2), "ehr_test_data",  tmp / "ehr"),
        "train_labels":      train_labels_path,
        "validation_labels": val_labels_path,
        "test_labels":       test_labels_path,
        "demographics_train": demo_train,
        "demographics_val":   demo_val,
        "demographics":       demo_test,
    }

    # ── Note ─────────────────────────────────────────────────────────────────
    note_files = {
        "train_data":        _resolve(up["note_train_data"], _make_note_data(n_train, 10), "note_train_data", tmp / "note"),
        "validation_data":   _resolve(up["note_val_data"],   _make_note_data(n_val,   11), "note_val_data",   tmp / "note"),
        "test_data":         _resolve(up["note_test_data"],  _make_note_data(n_test,  12), "note_test_data",  tmp / "note"),
        "train_labels":      train_labels_path,
        "validation_labels": val_labels_path,
        "test_labels":       test_labels_path,
    }

    # ── Image ─────────────────────────────────────────────────────────────────
    _synth_img = _make_image_data(n_train, tmp / "image", 20)
    image_files = {
        "train_data":        _resolve(up["image_train_data"], _synth_img,          "image_train_data", tmp / "image"),
        "validation_data":   _resolve(up["image_val_data"],   _synth_img[:n_val],  "image_val_data",   tmp / "image"),
        "test_data":         _resolve(up["image_test_data"],  _synth_img[:n_test], "image_test_data",  tmp / "image"),
        "train_labels":      train_labels_path,
        "validation_labels": val_labels_path,
        "test_labels":       test_labels_path,
    }

    file_paths_per_agent = {
        "ehr_agent":   ehr_files,
        "note_agent":  note_files,
        "image_agent": image_files,
    }
    return file_paths_per_agent, str(headers_file)


# ── Stage 1: DataAgent only (no LLM) ─────────────────────────────────────────

def stage1_data_agent(file_paths_per_agent, headers_file):
    """
    Directly call DataAgent._run_local() for each modality.
    No LLM / API key required.
    """
    print("=" * 60)
    print("Stage 1: DataAgent local mode (no LLM)")
    print("=" * 60)

    os.environ["CEREBRA_EHR_HEADER_PATH"] = headers_file

    # Import after setting env var
    from cerebra.agents.data_agent import DataAgent

    agent = DataAgent()

    for agent_name, fps in file_paths_per_agent.items():
        meta = agent._run_local(
            file_paths=fps,
            agent_name=agent_name,
            patient_id=PATIENT_IDX,
        )
        info = meta.get_metadata_info()
        status = info["status"]
        keys   = list(info["dataset"].keys())
        assert status == "success", f"{agent_name}: expected success, got {status}"
        print(f"  ✓ {agent_name:12s}  status={status}  keys={keys}")

    print("\nStage 1 PASSED\n")


# ── Stage 2: Full pipeline via run_cerebra.py ─────────────────────────────────

def stage2_full_pipeline(file_paths_per_agent, headers_file, output_json: str):
    """
    Call tasks/run_cerebra.py as a subprocess with synthetic local data.
    Requires a valid LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    """
    print("=" * 60)
    print("Stage 2: Full pipeline via tasks/run_cerebra.py")
    print("=" * 60)

    script = Path(__file__).parent.parent / "tasks" / "run_cerebra.py"
    if not script.exists():
        raise FileNotFoundError(f"run_cerebra.py not found at {script}")

    file_paths_json = json.dumps(file_paths_per_agent)

    cmd = [
        sys.executable, str(script),
        "--patient_id",  str(PATIENT_IDX),
        "--year",        str(YEAR),
        "--llm_engine",  "gpt-4o",
        "--output_json", output_json,
        "--file_paths",  file_paths_json,
        "--query",
        f"Predict dementia risk within {YEAR} year for patient {PATIENT_IDX}",
    ]

    env = {**os.environ, "CEREBRA_EHR_HEADER_PATH": headers_file}

    print("Command (truncated):")
    print(f"  python run_cerebra.py --patient_id {PATIENT_IDX} --year {YEAR} "
          f"--llm_engine gpt-4o --output_json <path> --file_paths <json>\n")

    result = subprocess.run(cmd, env=env, text=True)

    if result.returncode != 0:
        print(f"\n✗ Pipeline exited with code {result.returncode}")
        sys.exit(result.returncode)

    if Path(output_json).exists():
        with open(output_json) as f:
            out = json.load(f)
        print(f"\n✓ Output JSON written")
        print(f"  patient_id  : {out.get('patient_id')}")
        print(f"  agents_used : {out.get('agents_used')}")
        summary = out.get("prediction_result", {})
        if isinstance(summary, dict):
            for key in ("summary", "response", "description"):
                if key in summary:
                    print(f"  {key}: {str(summary[key])[:200]}")
                    break
    else:
        print("\n⚠  output_json not written "
              "(pipeline may have completed without it — check stdout above)")

    print("\nStage 2 DONE\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    with tempfile.TemporaryDirectory(prefix="cerebra_e2e_") as tmp:
        tmp_path = Path(tmp)
        print(f"\nTest data directory: {tmp_path}\n")

        file_paths_per_agent, headers_file = create_test_fixtures(tmp_path)

        # Print the --file_paths value for manual re-use
        print("── file_paths JSON (copy-paste into CLI) " + "─" * 20)
        print(json.dumps(file_paths_per_agent, indent=2))
        print()

        # ── Stage 1 ──────────────────────────────────────────────────────────
        stage1_data_agent(file_paths_per_agent, headers_file)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        output_json = str(tmp_path / "pipeline_output.json")
        stage2_full_pipeline(file_paths_per_agent, headers_file, output_json)


if __name__ == "__main__":
    main()
