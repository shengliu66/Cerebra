# cerebra/tools/data_agent/training_mode/tool.py
#
# Training mode (paper §Data Agent):
#   "Retrieves a complete cohort satisfying the criteria specified in the
#    natural-language query, including dementia cases and corresponding labels.
#    The retrieved data are then standardized for downstream model training and
#    split at the patient level into training, validation, and test sets to
#    prevent data leakage. Modality-specific representations are generated:
#    EHR data are encoded as SciPy sparse matrices, clinical notes are organised
#    as per-patient text sequences, and imaging data are referenced via paths to
#    pre-processed image files."

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cerebra.tools.base import BaseTool
from cerebra.tools.data_agent.postgresql_query_with_natural_language.tool import (
    PostgreSqlQueryWithNaturalLanguageTool,
)


class TrainingModeDataTool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string: str = "gpt-4o"):
        super().__init__(
            tool_name="TrainingModeDataTool",
            tool_description=(
                "Training mode: given a natural-language cohort description, queries the "
                "database for matching patients and their dementia outcome labels, splits "
                "at patient level into train/val/test sets (no leakage), and returns "
                "modality-specific representations: EHR as SciPy sparse matrices, notes "
                "as per-patient text sequences, and images as preprocessed file paths."
            ),
            tool_version="1.0.0",
            input_types={
                "postgresql_database_url": "str - SQLAlchemy URL for the PostgreSQL database",
                "natural_language_query": "str - Cohort inclusion criteria in natural language",
                "modalities": "list[str] - Modalities to retrieve: ehr, notes, images (default: all three)",
                "train_ratio": "float - Fraction of cohort assigned to training (default: 0.70)",
                "val_ratio": "float - Fraction assigned to validation (default: 0.15); remainder is test",
                "random_seed": "int - Seed for reproducible patient-level shuffling (default: 42)",
                "balance": "bool - Undersample majority class in training split (default: True)",
            },
            output_type=(
                "dict - {status, cohort_size, train, val, test (each with patient_ids, "
                "n_patients, labels, ehr/notes/images representations), split_metadata}"
            ),
            demo_commands=[
                {
                    "command": (
                        'tool.execute(postgresql_database_url="postgresql+psycopg2://user@host:5432/db", '
                        'natural_language_query="Patients over 65 with at least one T1 MRI scan ")'
                    ),
                    "description": "Retrieve training cohort with all three modalities.",
                },
                {
                    "command": (
                        'tool.execute(..., modalities=["ehr", "notes"], train_ratio=0.8, balance=False)'
                    ),
                    "description": "EHR + notes only, 80/10/10 split, no class balancing.",
                },
            ],
            user_metadata={
                "best_practices": [
                    "Split is at patient level to prevent data leakage across sets.",
                    "Set balance=True (default) to prevent class imbalance from skewing training.",
                    "Fix random_seed for reproducible experiments.",
                ]
            },
        )
        self._nl_tool = PostgreSqlQueryWithNaturalLanguageTool(model_string=model_string)

    # ── Prompt builders ────────────────────────────────────────────────────────

    @staticmethod
    def _cohort_prompt(natural_language_query: str) -> str:
        return (
            "[TRAINING MODE – cohort + labels]\n"
            "Retrieve all patients satisfying the following clinical criteria, together with "
            "their dementia outcome label (binary: 0 = no dementia, 1 = dementia).\n"
            "Return columns: pat_mrn_id, dementia_label, observation_date.\n"
            f"Criteria: {natural_language_query}"
        )

    @staticmethod
    def _ehr_prompt(natural_language_query: str) -> str:
        return (
            "[TRAINING MODE – EHR features]\n"
            "For the cohort described below, retrieve all EHR features (lab results, diagnoses, "
            "medications, vitals) per patient. Return pat_mrn_id plus all feature columns.\n"
            f"Cohort: {natural_language_query}"
        )

    @staticmethod
    def _notes_prompt(natural_language_query: str) -> str:
        return (
            "[TRAINING MODE – clinical notes]\n"
            "For the cohort described below, retrieve all clinical and radiology notes per patient. "
            "Return: pat_mrn_id, note_date, note_type, narrative, impression. "
            "Order by pat_mrn_id, note_date.\n"
            f"Cohort: {natural_language_query}"
        )

    @staticmethod
    def _images_prompt(natural_language_query: str) -> str:
        return (
            "[TRAINING MODE – MRI image paths]\n"
            "For the cohort described below, retrieve paths to preprocessed MRI image files. "
            "Return: pat_mrn_id, scan_date, scan_type, preprocessed_brain_img_path.\n"
            f"Cohort: {natural_language_query}"
        )

    # ── Modality-specific representation builders ──────────────────────────────

    @staticmethod
    def _ehr_representation(df: pd.DataFrame) -> Dict[str, Any]:
        """Per-patient SciPy CSR sparse matrix from EHR feature columns."""
        feature_cols = [c for c in df.columns if c != "pat_mrn_id"]
        matrices: Dict[str, sp.csr_matrix] = {}
        if "pat_mrn_id" in df.columns:
            for pid, grp in df.groupby("pat_mrn_id"):
                dense = grp[feature_cols].values.astype(float)
                matrices[str(pid)] = sp.csr_matrix(dense)
        return {"type": "sparse_matrix", "patient_matrices": matrices, "feature_columns": feature_cols}

    @staticmethod
    def _notes_representation(df: pd.DataFrame) -> Dict[str, Any]:
        """Per-patient list of concatenated note strings."""
        sequences: Dict[str, List[str]] = {}
        if "pat_mrn_id" not in df.columns:
            return {"type": "text_sequences", "patient_sequences": sequences}
        for pid, grp in df.groupby("pat_mrn_id"):
            texts = []
            for _, row in grp.iterrows():
                parts = []
                if "narrative" in row and pd.notna(row["narrative"]):
                    parts.append(f"Narrative: {row['narrative']}")
                if "impression" in row and pd.notna(row["impression"]):
                    parts.append(f"Impression: {row['impression']}")
                if parts:
                    texts.append("\n".join(parts))
            sequences[str(pid)] = texts
        return {"type": "text_sequences", "patient_sequences": sequences}

    @staticmethod
    def _images_representation(df: pd.DataFrame) -> Dict[str, Any]:
        """Per-patient list of preprocessed MRI file paths."""
        paths: Dict[str, List[str]] = {}
        if "pat_mrn_id" not in df.columns:
            return {"type": "image_paths", "patient_paths": paths}
        path_col = (
            "preprocessed_brain_img_path"
            if "preprocessed_brain_img_path" in df.columns
            else df.columns[-1]
        )
        for pid, grp in df.groupby("pat_mrn_id"):
            paths[str(pid)] = grp[path_col].dropna().tolist()
        return {"type": "image_paths", "patient_paths": paths}

    @staticmethod
    def _subset(representation: Dict[str, Any], patient_ids: List[str]) -> Dict[str, Any]:
        """Filter a representation to the given patient IDs."""
        key_map = {
            "sparse_matrix": "patient_matrices",
            "text_sequences": "patient_sequences",
            "image_paths": "patient_paths",
        }
        data_key = key_map.get(representation.get("type", ""), "patient_matrices")
        full = representation.get(data_key, {})
        subset = {pid: full[pid] for pid in patient_ids if pid in full}
        out = {k: v for k, v in representation.items() if k != data_key}
        out[data_key] = subset
        return out

    # ── Patient-level split ────────────────────────────────────────────────────

    @staticmethod
    def _split(
        patient_ids: List[str],
        labels: Dict[str, int],
        train_ratio: float,
        val_ratio: float,
        random_seed: int,
        balance: bool,
    ) -> Dict[str, List[str]]:
        rng = random.Random(random_seed)
        ids = patient_ids.copy()
        rng.shuffle(ids)
        if balance:
            pos = [p for p in ids if labels.get(p, 0) == 1]
            neg = [p for p in ids if labels.get(p, 0) == 0]
            n = min(len(pos), len(neg))
            rng.shuffle(pos); rng.shuffle(neg)
            ids = pos[:n] + neg[:n]
            rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return {
            "train": ids[:n_train],
            "val": ids[n_train: n_train + n_val],
            "test": ids[n_train + n_val:],
        }

    # ── Main execute ───────────────────────────────────────────────────────────

    def execute(
        self,
        postgresql_database_url: str,
        natural_language_query: str,
        modalities: Optional[List[str]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        balance: bool = True,
    ) -> Dict[str, Any]:
        if modalities is None:
            modalities = ["ehr", "notes", "images"]

        # ── Step 1: cohort + labels ──────────────────────────────────────────
        cohort_res = self._nl_tool.execute(
            postgresql_database_url=postgresql_database_url,
            natural_language_query=self._cohort_prompt(natural_language_query),
        )
        if cohort_res.get("status") != "success":
            return {"status": "error", "step": "cohort_retrieval", "detail": cohort_res}

        cohort_raw = cohort_res["data"].get_dataset()["dataset"]
        cohort_df = pd.DataFrame(cohort_raw) if isinstance(cohort_raw, dict) else cohort_raw

        id_col = "pat_mrn_id" if "pat_mrn_id" in cohort_df.columns else cohort_df.columns[0]
        patient_ids = [str(p) for p in cohort_df[id_col].tolist()]
        label_col = next(
            (c for c in cohort_df.columns if "dementia" in c.lower() or "label" in c.lower()),
            None,
        )
        labels: Dict[str, int] = {}
        if label_col:
            labels = {
                str(pid): int(lbl)
                for pid, lbl in zip(patient_ids, cohort_df[label_col].tolist())
            }

        # ── Step 2: modality retrieval ───────────────────────────────────────
        modality_reps: Dict[str, Any] = {}

        if "ehr" in modalities:
            r = self._nl_tool.execute(
                postgresql_database_url=postgresql_database_url,
                natural_language_query=self._ehr_prompt(natural_language_query),
            )
            if r.get("status") == "success":
                raw = r["data"].get_dataset()["dataset"]
                df = pd.DataFrame(raw) if isinstance(raw, dict) else raw
                modality_reps["ehr"] = self._ehr_representation(df)

        if "notes" in modalities:
            r = self._nl_tool.execute(
                postgresql_database_url=postgresql_database_url,
                natural_language_query=self._notes_prompt(natural_language_query),
            )
            if r.get("status") == "success":
                raw = r["data"].get_dataset()["dataset"]
                df = pd.DataFrame(raw) if isinstance(raw, dict) else raw
                modality_reps["notes"] = self._notes_representation(df)

        if "images" in modalities:
            r = self._nl_tool.execute(
                postgresql_database_url=postgresql_database_url,
                natural_language_query=self._images_prompt(natural_language_query),
            )
            if r.get("status") == "success":
                raw = r["data"].get_dataset()["dataset"]
                df = pd.DataFrame(raw) if isinstance(raw, dict) else raw
                modality_reps["images"] = self._images_representation(df)

        # ── Step 3: patient-level split ──────────────────────────────────────
        splits = self._split(patient_ids, labels, train_ratio, val_ratio, random_seed, balance)

        # ── Step 4: assemble output ──────────────────────────────────────────
        output: Dict[str, Any] = {"status": "success", "cohort_size": len(patient_ids)}
        for split_name, split_ids in splits.items():
            output[split_name] = {
                "patient_ids": split_ids,
                "n_patients": len(split_ids),
                "labels": {pid: labels[pid] for pid in split_ids if pid in labels},
                **{mod: self._subset(rep, split_ids) for mod, rep in modality_reps.items()},
            }
        output["split_metadata"] = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 4),
            "random_seed": random_seed,
            "balanced": balance,
            "modalities_retrieved": list(modality_reps.keys()),
        }
        return output
