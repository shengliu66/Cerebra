# cerebra/tools/data_agent/exploration_mode/tool.py
#
# Exploration mode (paper §Data Agent):
#   "The exploration mode enables ad hoc database interrogation through the
#    user dashboard. This mode supports tasks such as reviewing patient medical
#    histories under specific constraints, identifying cohorts with similar
#    clinical profiles, and inspecting database schemas and metadata,
#    facilitating both hypothesis generation and data understanding."

import pandas as pd
from typing import Any, Dict, List, Optional

from cerebra.engine.factory import create_llm_engine
from cerebra.tools.base import BaseTool
from cerebra.tools.data_agent.postgresql_query_with_natural_language.tool import (
    PostgreSqlQueryWithNaturalLanguageTool,
)

EXPLORATION_TYPES = ("schema", "cohort", "patient_history", "similar_patients")


class ExplorationModeDataTool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string: str = "gpt-4o"):
        super().__init__(
            tool_name="ExplorationModeDataTool",
            tool_description=(
                "Exploration mode: ad hoc database interrogation for hypothesis generation "
                "and data understanding. Supports four exploration types:\n"
                "  • schema           – inspect database tables, columns, and types\n"
                "  • cohort           – find patients matching NL criteria + summary statistics\n"
                "  • patient_history  – review complete history of a specific patient\n"
                "  • similar_patients – identify patients with similar clinical profiles"
            ),
            tool_version="1.0.0",
            input_types={
                "postgresql_database_url": "str - SQLAlchemy URL for the PostgreSQL database",
                "natural_language_query": "str - Free-form question or criteria description",
                "exploration_type": (
                    "str - One of: schema | cohort | patient_history | similar_patients "
                    "(default: cohort)"
                ),
                "patient_id": "str (optional) - Required for patient_history exploration type",
            },
            output_type=(
                "dict - {status, exploration_type, + type-specific fields: "
                "raw_schema/summary for schema; n_patients/cohort_summary/data_preview for cohort; "
                "n_records/timeline/date_range for patient_history; n_matches/results for similar_patients}"
            ),
            demo_commands=[
                {
                    "command": 'tool.execute(..., natural_language_query="", exploration_type="schema")',
                    "description": "Inspect all database tables and columns with an LLM-generated summary.",
                },
                {
                    "command": (
                        'tool.execute(..., natural_language_query="Patients over 70 with MCI diagnosis", '
                        'exploration_type="cohort")'
                    ),
                    "description": "Discover cohort matching clinical criteria + summary statistics.",
                },
                {
                    "command": (
                        'tool.execute(..., natural_language_query="", exploration_type="patient_history", '
                        'patient_id="MRN_12345")'
                    ),
                    "description": "Review complete medical history for a specific patient.",
                },
                {
                    "command": (
                        'tool.execute(..., natural_language_query="Female, age 75-85, APOE4 carrier, '
                        'memory complaints", exploration_type="similar_patients")'
                    ),
                    "description": "Find patients with similar clinical profile for comparative review.",
                },
            ],
            user_metadata={
                "best_practices": [
                    "Use 'schema' first when unfamiliar with the database structure.",
                    "Use 'cohort' to size and characterise a patient population before training.",
                    "'similar_patients' is useful for case-based reasoning and sanity checks.",
                ]
            },
        )
        self._nl_tool = PostgreSqlQueryWithNaturalLanguageTool(model_string=model_string)
        self._llm = create_llm_engine(model_string=model_string, is_multimodal=False)

    # ── Schema exploration ─────────────────────────────────────────────────────

    def _explore_schema(self, postgresql_database_url: str) -> Dict[str, Any]:
        from sqlalchemy import create_engine
        engine = create_engine(postgresql_database_url)
        raw_schema = PostgreSqlQueryWithNaturalLanguageTool._generate_pg_schema_for_llm(engine)

        summary_prompt = (
            f"Given this clinical database schema:\n\n{raw_schema}\n\n"
            "Write a concise, plain-language summary (3-6 sentences) aimed at clinical researchers. "
            "Describe: what patient data is stored, what each major table represents, "
            "and what kinds of clinical queries or analyses this database supports."
        )
        summary = self._llm(prompt=summary_prompt)

        return {
            "exploration_type": "schema",
            "raw_schema": raw_schema,
            "summary": summary if isinstance(summary, str) else str(summary),
        }

    # ── Cohort discovery ───────────────────────────────────────────────────────

    def _explore_cohort(
        self, natural_language_query: str, postgresql_database_url: str
    ) -> Dict[str, Any]:
        enriched = (
            "[EXPLORATION MODE – cohort discovery]\n"
            f"Find all patients satisfying: {natural_language_query}.\n"
            "Return: pat_mrn_id, age (if available), gender (if available), "
            "dementia_label (if available), observation_date. "
            "Include as many descriptive fields as the schema allows."
        )
        r = self._nl_tool.execute(
            postgresql_database_url=postgresql_database_url,
            natural_language_query=enriched,
        )
        if r.get("status") != "success":
            return {"status": "error", "detail": r}

        raw = r["data"].get_dataset()["dataset"]
        df = pd.DataFrame(raw) if isinstance(raw, dict) else raw

        stats: Dict[str, Any] = {"n_patients": len(df)}
        for label_col in [c for c in df.columns if "dementia" in c.lower() or "label" in c.lower()]:
            stats["dementia_positive"] = int((df[label_col] == 1).sum())
            stats["dementia_negative"] = int((df[label_col] == 0).sum())
            break
        if "age" in df.columns:
            stats["age_mean"] = round(float(df["age"].mean()), 1)
            stats["age_std"] = round(float(df["age"].std()), 1)
        if "gender" in df.columns:
            stats["gender_distribution"] = df["gender"].value_counts().to_dict()

        return {
            "exploration_type": "cohort",
            "n_patients": len(df),
            "cohort_summary": stats,
            "patient_ids": df["pat_mrn_id"].tolist() if "pat_mrn_id" in df.columns else [],
            "data_preview": df.head(10).to_dict("records"),
            "sql_query": r.get("sql_query"),
        }

    # ── Patient history review ─────────────────────────────────────────────────

    def _explore_patient_history(
        self, patient_id: str, natural_language_query: str, postgresql_database_url: str
    ) -> Dict[str, Any]:
        enriched = (
            "[EXPLORATION MODE – patient history review]\n"
            f"Retrieve the complete medical history for patient pat_mrn_id = '{patient_id}'. "
            "Include all available data: diagnoses, lab results, medications, procedures, "
            "clinical notes, radiology reports, imaging studies. "
            "Order all results chronologically."
        )
        if natural_language_query:
            enriched += f"\nAdditional context or constraint: {natural_language_query}"

        r = self._nl_tool.execute(
            postgresql_database_url=postgresql_database_url,
            natural_language_query=enriched,
        )
        if r.get("status") != "success":
            return {"status": "error", "detail": r}

        raw = r["data"].get_dataset()["dataset"]
        df = pd.DataFrame(raw) if isinstance(raw, dict) else raw
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df = df.sort_values(date_col, na_position="last")

        return {
            "exploration_type": "patient_history",
            "patient_id": patient_id,
            "n_records": len(df),
            "timeline": df.to_dict("records"),
            "date_range": {
                "earliest": str(df[date_col].min()) if date_col else None,
                "latest": str(df[date_col].max()) if date_col else None,
            },
            "sql_query": r.get("sql_query"),
        }

    # ── Similar patient search ─────────────────────────────────────────────────

    def _explore_similar_patients(
        self, natural_language_query: str, postgresql_database_url: str
    ) -> Dict[str, Any]:
        enriched = (
            "[EXPLORATION MODE – similar patient search]\n"
            f"Find patients whose clinical profile matches: {natural_language_query}.\n"
            "Return: pat_mrn_id plus all clinically relevant matching features "
            "(age, gender, diagnoses, lab values, medication history). "
            "Include every patient that satisfies the criteria."
        )
        r = self._nl_tool.execute(
            postgresql_database_url=postgresql_database_url,
            natural_language_query=enriched,
        )
        if r.get("status") != "success":
            return {"status": "error", "detail": r}

        raw = r["data"].get_dataset()["dataset"]
        df = pd.DataFrame(raw) if isinstance(raw, dict) else raw

        return {
            "exploration_type": "similar_patients",
            "n_matches": len(df),
            "matching_patient_ids": df["pat_mrn_id"].tolist() if "pat_mrn_id" in df.columns else [],
            "results": df.to_dict("records"),
            "sql_query": r.get("sql_query"),
        }

    # ── Main execute ───────────────────────────────────────────────────────────

    def execute(
        self,
        postgresql_database_url: str,
        natural_language_query: str,
        exploration_type: str = "cohort",
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if exploration_type not in EXPLORATION_TYPES:
            return {
                "status": "error",
                "message": (
                    f"Unknown exploration_type '{exploration_type}'. "
                    f"Choose from: {', '.join(EXPLORATION_TYPES)}"
                ),
            }

        if exploration_type == "schema":
            result = self._explore_schema(postgresql_database_url)
        elif exploration_type == "cohort":
            result = self._explore_cohort(natural_language_query, postgresql_database_url)
        elif exploration_type == "patient_history":
            if not patient_id:
                return {
                    "status": "error",
                    "message": "patient_id is required for exploration_type='patient_history'",
                }
            result = self._explore_patient_history(
                patient_id, natural_language_query, postgresql_database_url
            )
        else:  # similar_patients
            result = self._explore_similar_patients(natural_language_query, postgresql_database_url)

        result.setdefault("status", "success")
        return result
