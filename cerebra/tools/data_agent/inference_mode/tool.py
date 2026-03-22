# cerebra/tools/data_agent/inference_mode/tool.py
#
# Inference mode (paper §Data Agent):
#   "Retrieves the longitudinal medical history of an individual patient,
#    optionally constrained by temporal or clinical criteria specified in the
#    query. The data are processed using the same modality-specific pipelines
#    as in training mode, but without associated outcome labels."

import pandas as pd
from typing import Any, Dict, List, Optional

from cerebra.tools.base import BaseTool
from cerebra.tools.data_agent.postgresql_query_with_natural_language.tool import (
    PostgreSqlQueryWithNaturalLanguageTool,
)


class InferenceModeDataTool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string: str = "gpt-4o"):
        super().__init__(
            tool_name="InferenceModeDataTool",
            tool_description=(
                "Inference mode: retrieves the complete longitudinal medical history for a "
                "single patient, optionally constrained by temporal or clinical criteria. "
                "Applies the same modality-specific pipelines as training mode (EHR features, "
                "note text sequences, MRI file paths) but does NOT return any outcome labels."
            ),
            tool_version="1.0.0",
            input_types={
                "postgresql_database_url": "str - SQLAlchemy URL for the PostgreSQL database",
                "patient_id": "str - pat_mrn_id of the patient to retrieve",
                "natural_language_constraint": (
                    "str (optional) - Temporal or clinical constraint, "
                    "e.g. 'only records from the past 3 years' or 'exclude imaging before 2020'"
                ),
                "modalities": "list[str] - Modalities to retrieve: ehr, notes, images (default: all three)",
            },
            output_type=(
                "dict - {status, patient_id, has_outcome_labels=False, timeline (chronological events), "
                "n_timeline_events, constraint_applied, ehr (optional), notes (optional), images (optional)}"
            ),
            demo_commands=[
                {
                    "command": (
                        'tool.execute(postgresql_database_url="postgresql+psycopg2://user@host:5432/db", '
                        'patient_id="MRN_12345")'
                    ),
                    "description": "Retrieve full longitudinal history for a single patient.",
                },
                {
                    "command": (
                        'tool.execute(..., patient_id="MRN_12345", '
                        'natural_language_constraint="only records from the past 2 years", '
                        'modalities=["notes"])'
                    ),
                    "description": "Notes only, time-constrained retrieval.",
                },
            ],
            user_metadata={
                "best_practices": [
                    "has_outcome_labels is always False — use training mode when labels are needed.",
                    "Timeline is sorted chronologically by the earliest available date column.",
                    "Use natural_language_constraint to narrow to a clinically relevant window.",
                ]
            },
        )
        self._nl_tool = PostgreSqlQueryWithNaturalLanguageTool(model_string=model_string)

    # ── Prompt builders ────────────────────────────────────────────────────────

    @staticmethod
    def _timeline_prompt(patient_id: str, constraint: str) -> str:
        prompt = (
            f"[INFERENCE MODE – longitudinal timeline]\n"
            f"Retrieve the complete chronological medical history for patient pat_mrn_id = '{patient_id}'. "
            f"Include all event types: diagnoses, lab results, medications, procedures, notes, imaging. "
            f"Return: event_date, event_type, description, value, units. "
            f"Order results by event_date ascending. "
            f"Do NOT include any dementia outcome labels."
        )
        if constraint:
            prompt += f"\nConstraint: {constraint}"
        return prompt

    @staticmethod
    def _ehr_prompt(patient_id: str, constraint: str) -> str:
        prompt = (
            f"[INFERENCE MODE – EHR features]\n"
            f"Retrieve all EHR features (labs, diagnoses, medications, vitals) for patient "
            f"pat_mrn_id = '{patient_id}', ordered by date. Do NOT include dementia labels."
        )
        if constraint:
            prompt += f"\nConstraint: {constraint}"
        return prompt

    @staticmethod
    def _notes_prompt(patient_id: str, constraint: str) -> str:
        prompt = (
            f"[INFERENCE MODE – clinical notes]\n"
            f"Retrieve all clinical and radiology notes for patient pat_mrn_id = '{patient_id}'. "
            f"Return: note_date, note_type, narrative, impression. Order by note_date ascending. "
            f"Do NOT include dementia labels."
        )
        if constraint:
            prompt += f"\nConstraint: {constraint}"
        return prompt

    @staticmethod
    def _images_prompt(patient_id: str, constraint: str) -> str:
        prompt = (
            f"[INFERENCE MODE – MRI image paths]\n"
            f"Retrieve preprocessed MRI image paths for patient pat_mrn_id = '{patient_id}'. "
            f"Return: scan_date, scan_type, preprocessed_brain_img_path."
        )
        if constraint:
            prompt += f"\nConstraint: {constraint}"
        return prompt

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df = df.sort_values(date_col, na_position="last")
        return df

    @staticmethod
    def _notes_sequence(df: pd.DataFrame) -> List[str]:
        texts = []
        for _, row in df.iterrows():
            parts = []
            if "narrative" in row and pd.notna(row["narrative"]):
                parts.append(f"Narrative: {row['narrative']}")
            if "impression" in row and pd.notna(row["impression"]):
                parts.append(f"Impression: {row['impression']}")
            if parts:
                texts.append("\n".join(parts))
        return texts

    # ── Main execute ───────────────────────────────────────────────────────────

    def execute(
        self,
        postgresql_database_url: str,
        patient_id: str,
        natural_language_constraint: Optional[str] = None,
        modalities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if modalities is None:
            modalities = ["ehr", "notes", "images"]
        constraint = natural_language_constraint or ""

        # ── Step 1: chronological timeline ──────────────────────────────────
        timeline_res = self._nl_tool.execute(
            postgresql_database_url=postgresql_database_url,
            natural_language_query=self._timeline_prompt(patient_id, constraint),
        )
        if timeline_res.get("status") != "success":
            return {"status": "error", "step": "timeline_retrieval", "detail": timeline_res}

        raw = timeline_res["data"].get_dataset()["dataset"]
        timeline_df = self._sort_by_date(pd.DataFrame(raw) if isinstance(raw, dict) else raw)

        out: Dict[str, Any] = {
            "status": "success",
            "patient_id": patient_id,
            "has_outcome_labels": False,   # explicit invariant for inference mode
            "timeline": timeline_df.to_dict("records"),
            "n_timeline_events": len(timeline_df),
            "constraint_applied": constraint or None,
        }

        # ── Step 2: modality data ────────────────────────────────────────────
        if "ehr" in modalities:
            r = self._nl_tool.execute(
                postgresql_database_url=postgresql_database_url,
                natural_language_query=self._ehr_prompt(patient_id, constraint),
            )
            if r.get("status") == "success":
                raw = r["data"].get_dataset()["dataset"]
                df = self._sort_by_date(pd.DataFrame(raw) if isinstance(raw, dict) else raw)
                out["ehr"] = {"data": df.to_dict("records"), "n_records": len(df), "type": "ehr_features"}

        if "notes" in modalities:
            r = self._nl_tool.execute(
                postgresql_database_url=postgresql_database_url,
                natural_language_query=self._notes_prompt(patient_id, constraint),
            )
            if r.get("status") == "success":
                raw = r["data"].get_dataset()["dataset"]
                df = self._sort_by_date(pd.DataFrame(raw) if isinstance(raw, dict) else raw)
                texts = self._notes_sequence(df)
                out["notes"] = {"text_sequence": texts, "n_notes": len(texts), "type": "text_sequences"}

        if "images" in modalities:
            r = self._nl_tool.execute(
                postgresql_database_url=postgresql_database_url,
                natural_language_query=self._images_prompt(patient_id, constraint),
            )
            if r.get("status") == "success":
                raw = r["data"].get_dataset()["dataset"]
                df = self._sort_by_date(pd.DataFrame(raw) if isinstance(raw, dict) else raw)
                path_col = (
                    "preprocessed_brain_img_path"
                    if "preprocessed_brain_img_path" in df.columns
                    else (df.columns[-1] if not df.empty else None)
                )
                out["images"] = {
                    "file_paths": df[path_col].dropna().tolist() if path_col else [],
                    "type": "image_paths",
                }

        return out
