# agents/data_agent.py
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from cerebra.agents.lightweight_agent import LightweightAgent
from cerebra.utils.ehr_headers import load_ehr_headers
from cerebra.utils.metadata import Metadata
from cerebra.utils.utils import save_to_pickle

load_dotenv(find_dotenv())
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")
MODES = ("training", "inference", "exploration", "local")

logger = logging.getLogger(__name__)


class DataAgent(LightweightAgent):
    """
    Data agent with four operational modes:

      "training"   — SQL/NL: retrieve cohort + labels from a PostgreSQL database,
                     split at patient level into train/val/test, return modality
                     representations ready for model training.

      "inference"  — SQL/NL: retrieve longitudinal history for a single patient
                     from the database, without outcome labels.

      "exploration"— SQL/NL: ad hoc database interrogation (schema inspection,
                     cohort discovery, patient history review, similar-patient search).

      "local"      — File-based: load pre-processed pickle/CSV files from disk via
                     a caller-supplied dict of file paths. No database required.
    """

    def __init__(self, llm_engine_name: str = "gpt-4o"):
        super().__init__(
            agent_name="data_agent",
            llm_engine_name=llm_engine_name,
            enabled_tools=["training_mode", "inference_mode", "exploration_mode"],
            verbose=True,
        )
        # SQL-backed mode tools — imported and constructed lazily on first use
        self._sql_tools: Dict[str, Any] = {}
        self.metadata: Optional[Metadata] = None

    # ── Mode tool lazy loader ─────────────────────────────────────────────────

    def _get_sql_tool(self, mode: str):
        if mode not in self._sql_tools:
            if mode == "training":
                from cerebra.tools.data_agent.training_mode.tool import TrainingModeDataTool
                self._sql_tools[mode] = TrainingModeDataTool(self.llm_engine_name)
            elif mode == "inference":
                from cerebra.tools.data_agent.inference_mode.tool import InferenceModeDataTool
                self._sql_tools[mode] = InferenceModeDataTool(self.llm_engine_name)
            elif mode == "exploration":
                from cerebra.tools.data_agent.exploration_mode.tool import ExplorationModeDataTool
                self._sql_tools[mode] = ExplorationModeDataTool(self.llm_engine_name)
        return self._sql_tools[mode]

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, mode: str, **kwargs) -> Metadata:
        """
        Run the data agent in one of four modes.

        Args:
            mode: One of "training" | "inference" | "exploration" | "local"
            **kwargs:
                training:
                    postgresql_database_url (str)
                    natural_language_query  (str)
                    modalities  (list, default all)
                    train_ratio (float, default 0.70)
                    val_ratio   (float, default 0.15)
                    random_seed (int,   default 42)
                    balance     (bool,  default True)

                inference:
                    postgresql_database_url     (str)
                    patient_id                  (str, required)
                    natural_language_constraint (str, optional)
                    modalities                  (list, default all)

                exploration:
                    postgresql_database_url (str)
                    natural_language_query  (str)
                    exploration_type        (str) – schema | cohort | patient_history | similar_patients
                    patient_id              (str, required for patient_history)

                local:
                    file_paths  (dict) – maps split/role name → file path, e.g.
                                  { "train_data": "...", "test_data": "...",
                                    "train_labels": "...", "test_labels": "...",
                                    "demographics": "..." }
                    agent_name  (str)  – "ehr_agent" | "note_agent" | "image_agent"
                    patient_id  (int, optional) – index into test split

        Returns:
            Metadata for all modes.
        """
        if mode not in MODES:
            return {"status": "error", "message": f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}"}

        if mode == "local":
            return self._run_local(
                file_paths=kwargs.get("file_paths", {}),
                agent_name=kwargs.get("agent_name"),
                patient_id=kwargs.get("patient_id"),
            )

        # SQL modes
        tool = self._get_sql_tool(mode)

        if mode == "inference":
            patient_id = kwargs.pop("patient_id", None)
            if not patient_id:
                return {"status": "error", "message": "patient_id is required for inference mode"}
            result = tool.execute(
                patient_id=patient_id,
                natural_language_constraint=kwargs.pop("natural_language_query", None),
                **kwargs,
            )
        else:
            result = tool.execute(**kwargs)

        return self._wrap_sql_result(result, mode)

    # ── SQL result → Metadata wrapper ────────────────────────────────────────

    def _wrap_sql_result(self, result: Dict[str, Any], mode: str) -> Metadata:
        """Wrap a SQL-mode result dict in a Metadata object for consistency."""
        status = result.get("status", "success")
        cache_dir = os.path.join(CACHE_DIR, "data_agent", mode)
        os.makedirs(cache_dir, exist_ok=True)

        # Flatten the result into the dataset structure expected by Metadata
        dataset: Dict[str, Any] = {}
        for key, value in result.items():
            if key == "status":
                continue
            saved_path = save_to_pickle(value, key, cache_dir, check_exist=False)
            dataset[key] = {
                "saved_path": saved_path,
                "description": f"{mode} mode output: {key}",
                "configuration": {
                    "num_samples": len(value) if hasattr(value, "__len__") else "unknown"
                },
            }

        self.metadata = Metadata.create_agent_output(
            status=status,
            dataset=dataset,
            model={},
            cache_directory=os.path.join(CACHE_DIR, "data_agent"),
            agent_name="data_agent",
        )
        self.metadata.save()
        return self.metadata

    # ── Local (file-based) mode ───────────────────────────────────────────────

    def _run_local(
        self,
        file_paths: Dict[str, str],
        agent_name: str,
        patient_id: Optional[int] = None,
    ) -> Metadata:
        """Load pre-processed data files and return as Metadata."""

        # 1. Load all files except demographics (handled separately)
        loaded: Dict[str, Any] = {
            key: self._load_data_file(path)
            for key, path in file_paths.items()
            if key != "demographics"
        }

        # 2. Agent-specific preprocessing
        if agent_name == "note_agent" and loaded.get("train_data") is not None:
            loaded["train_data"] = self._truncate_notes(loaded["train_data"])

        # 3. Filter test split to a single patient if requested
        if patient_id is not None:
            patient_idx = int(patient_id)
            if loaded.get("test_data") is not None:
                test = loaded["test_data"]
                loaded["test_data"] = (
                    test.iloc[[patient_idx]] if isinstance(test, pd.DataFrame)
                    else [test[patient_idx]]
                )
            if loaded.get("test_labels") is not None:
                loaded["test_labels"] = [loaded["test_labels"][patient_idx]]

        # 4. Log sample counts
        for split in ("train_labels", "validation_labels", "test_labels"):
            if loaded.get(split) is not None:
                logger.info("# %s samples: %d", split.replace("_labels", ""), len(loaded[split]))

        # 5. Optional demographics
        demographic_info = None
        if "demographics" in file_paths and patient_id is not None:
            demo_list = self._load_data_file(file_paths["demographics"])
            if demo_list is not None:
                demographic_info = self.get_demographic_info(demo_list, patient_idx)

        # 6. Raw data for downstream use
        raw_data = None
        if patient_id is not None and file_paths.get("test_data"):
            test_raw = self._load_data_file(file_paths["test_data"])
            if test_raw is not None:
                if agent_name == "ehr_agent":
                    raw_data = self.get_ehr_nonzero_features(test_raw[patient_idx])
                elif agent_name == "note_agent":
                    raw_data = self.get_raw_notes_with_indices(test_raw[patient_idx])
                elif agent_name == "image_agent":
                    raw_data = test_raw

        # 7. Save and build output dataset
        feature_descriptions = self._get_feature_descriptions(agent_name)
        cache_dir = os.path.join(CACHE_DIR, "data_agent", agent_name or "unknown")
        os.makedirs(cache_dir, exist_ok=True)

        saved_paths: Dict[str, str] = {}
        for key, value in loaded.items():
            key_with_prefix = (
                f"patient_{patient_id}_{key}"
                if patient_id is not None and "test" in key
                else key
            )
            saved_paths[key] = save_to_pickle(
                value, key_with_prefix, cache_dir, check_exist=("test" not in key)
            )

        if demographic_info is not None:
            saved_paths["demographic_info"] = save_to_pickle(
                demographic_info, f"patient_{patient_id}_demographic_info", cache_dir, check_exist=False
            )
        if raw_data is not None:
            saved_paths["raw_data"] = save_to_pickle(
                raw_data, f"patient_{patient_id}_raw_data", cache_dir, check_exist=False
            )

        output_dataset = {
            k: {
                "saved_path": saved_paths[k],
                "description": feature_descriptions.get(k, ""),
                "configuration": {
                    "num_samples": len(loaded[k]) if k in loaded and hasattr(loaded[k], "__len__") else "unknown"
                },
            }
            for k in saved_paths
        }

        self.metadata = Metadata.create_agent_output(
            status="success",
            dataset=output_dataset,
            model={},
            cache_directory=os.path.join(CACHE_DIR, "data_agent"),
            agent_name="data_agent",
        )
        self.metadata.save()
        logger.info("Metadata saved to %s", self.metadata.metadata_config_path)
        return self.metadata

    # ── File-loading helpers ──────────────────────────────────────────────────

    def _load_data_file(self, path: str):
        """Load a pickle or CSV file. Returns None if the path does not exist."""
        if not path or not os.path.exists(path):
            return None
        if path.endswith(".csv"):
            return pd.read_csv(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def _truncate_notes(self, data: list, max_sentences: int = 2048) -> list:
        """Truncate each note to keep only the last max_sentences sentences."""
        for i, patient_data in enumerate(data):
            for j, text in enumerate(patient_data):
                sentences = text.replace("..", "").split(".")
                if len(sentences) > max_sentences:
                    data[i][j] = ".".join(sentences[-max_sentences:])
        return data

    def _get_feature_descriptions(self, agent_name: str) -> dict:
        """Retrieve data_descriptions from the modality agent class, then add shared label keys.

        Each modality agent (EHRAgent, NoteAgent, ImageAgent, ...) owns its own
        data_descriptions class attribute. DataAgent discovers it dynamically so
        new modalities require no changes here.
        """
        import importlib
        base: dict = {}
        if agent_name:
            try:
                mod = importlib.import_module(f"cerebra.agents.{agent_name}")
                for attr in vars(mod).values():
                    if isinstance(attr, type) and hasattr(attr, "data_descriptions"):
                        base = dict(attr.data_descriptions)
                        break
            except ModuleNotFoundError:
                logger.warning("No agent module found for '%s'; data descriptions will be empty.", agent_name)
        base.update({
            "train_labels":      "Labels for training",
            "validation_labels": "Labels for validation",
            "test_labels":       "Labels for testing",
        })
        return base

    # ── Patient-level helpers (used by both local mode and downstream agents) ─

    def get_demographic_info(self, demo_list: list, patient_id: int) -> Dict[str, Any]:
        """Extract structured demographic info from a loaded demographics list."""
        info = demo_list[patient_id]

        apoe_keys = [k for k in info if "APOE" in k]
        apoe_values = [info[k] for k in apoe_keys]
        if any(v > 0 for v in apoe_values):
            apoe_status = True
        elif any(v == 0 for v in apoe_values):
            apoe_status = False
        else:
            apoe_status = "Unknown"

        memory_keys = [k for k in info if "Memory" in k or "memory" in k]
        memory_values = [info[k] for k in memory_keys]
        if any(v > 0 for v in memory_values):
            memory_loss = True
        elif any(v == 0 for v in memory_values):
            memory_loss = False
        else:
            memory_loss = "Unknown"

        gender = info.get("gender", "Unknown")
        if isinstance(gender, str) and "demographics_GENDER_" in gender:
            gender = gender.replace("demographics_GENDER_", "")

        return {
            "APOE": apoe_status,
            "Age": info.get("age", "Unknown"),
            "Gender": gender,
            "Memory Loss": memory_loss,
        }

    def get_raw_notes_with_indices(self, note_data: List[str]) -> Dict[str, Any]:
        """Get raw notes with paragraph indices for traceability."""
        indexed_notes = []
        for note_idx, note in enumerate(note_data):
            paragraphs = note.split("\n\n") if "\n\n" in note else [note]
            note_entry = {
                "note_index": note_idx,
                "full_text": note,
                "paragraphs": [
                    {
                        "paragraph_index": para_idx,
                        "text": para.strip(),
                        "char_start": note.find(para),
                        "char_end": note.find(para) + len(para),
                    }
                    for para_idx, para in enumerate(paragraphs)
                    if para.strip()
                ],
            }
            indexed_notes.append(note_entry)
        return {
            "indexed_notes": indexed_notes,
            "total_notes": len(note_data),
            "total_paragraphs": sum(len(n["paragraphs"]) for n in indexed_notes),
        }

    def get_ehr_nonzero_features(self, ehr_data) -> str:
        """Return non-zero EHR features for a patient as an XML string."""
        ehr_header_names = load_ehr_headers()
        if len(ehr_data.shape) > 1:
            patient_vector = ehr_data.max(axis=0).toarray().flatten()
        else:
            patient_vector = ehr_data.toarray().flatten() if hasattr(ehr_data, "toarray") else ehr_data
        nonzero_indices = np.nonzero(patient_vector)[0]
        xml_parts = ["<ehr_features>"]
        for idx in nonzero_indices:
            name = ehr_header_names[idx] if idx < len(ehr_header_names) else f"feature_{idx}"
            xml_parts.append(f'  <feature name="{name}" value="{patient_vector[idx]}"/>')
        xml_parts.append("</ehr_features>")
        return "\n".join(xml_parts)

    # ── Agent registry ────────────────────────────────────────────────────────

    def register_agent_capabilities(self) -> Dict[str, Any]:
        return {
            "data_agent": {
                "agent_description": "Data agent with four modes: training, inference, exploration, local",
                "agent_capabilities": [
                    "training: SQL/NL cohort retrieval + labels, patient-level train/val/test split, modality representations",
                    "inference: SQL/NL single-patient longitudinal history retrieval, no outcome labels",
                    "exploration: SQL/NL schema inspection, cohort discovery, patient history, similar-patient search",
                    "local: load pre-processed pickle/CSV files from disk via a file_paths dict",
                ],
                "agent_input_types": {
                    "mode": "str – training | inference | exploration | local",
                    "**kwargs": "mode-specific arguments (see DataAgent.run docstring)",
                },
                "agent_output_type": "Metadata (local mode) or Dict (SQL modes)",
            }
        }
