"""Unit tests for DataAgent — local mode, helpers, and new infrastructure.

The test stubs out the heavy LLM/engine dependency chain so that running
these tests requires only the standard library + numpy/pandas/scipy.
"""
import os
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight stub for LightweightAgent and its transitive dependencies
# ---------------------------------------------------------------------------
def _stub_modules():
    """Insert minimal stubs into sys.modules before the real imports fire."""
    stubs = [
        "dotenv",
        "llm_output_parser",
        "openai", "anthropic", "together",
        "cerebra.engine",
        "cerebra.engine.factory",
        "cerebra.agents.lightweight_agent",
        "cerebra.agents.modules",
        "cerebra.agents.modules.initializer",
        "cerebra.agents.modules.planner",
        "cerebra.agents.modules.formatters",
        "pydantic",
        "nibabel",
        "PIL", "PIL.Image",
    ]
    for name in stubs:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    dotenv = sys.modules["dotenv"]
    dotenv.find_dotenv = lambda *a, **kw: ""
    dotenv.load_dotenv = lambda *a, **kw: None

    sys.modules["PIL.Image"].open = MagicMock()
    sys.modules["PIL"].Image = sys.modules["PIL.Image"]

    class _LightweightAgent:
        def __init__(self, agent_name="", llm_engine_name="gpt-4o",
                     enabled_tools=None, verbose=False):
            self.agent_name = agent_name
            self.llm_engine_name = llm_engine_name

    la_mod = sys.modules["cerebra.agents.lightweight_agent"]
    la_mod.LightweightAgent = _LightweightAgent


_stub_modules()

from cerebra.agents.data_agent import DataAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# 1. Local mode — core pipeline
# ---------------------------------------------------------------------------
class TestDataAgentLocalMode(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        patcher = patch("cerebra.agents.data_agent.CACHE_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.agent = DataAgent()

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def _run(self, file_paths, agent_name, patient_id=None):
        return self.agent.run(mode="local", file_paths=file_paths,
                               agent_name=agent_name, patient_id=patient_id)

    def test_unknown_mode_returns_error(self):
        result = self.agent.run(mode="bogus")
        self.assertEqual(result["status"], "error")
        self.assertIn("bogus", result["message"])

    def test_all_known_modes_not_rejected(self):
        """'local' is accepted; SQL modes are also in MODES (fail later, not here)."""
        from cerebra.agents.data_agent import MODES
        self.assertIn("local", MODES)
        self.assertIn("training", MODES)
        self.assertIn("inference", MODES)
        self.assertIn("exploration", MODES)

    def test_ehr_agent_returns_success_metadata(self):
        import scipy.sparse as sp, numpy as np
        train_data   = [sp.csr_matrix(np.eye(5)) for _ in range(3)]
        train_labels = [0, 1, 0]
        test_data    = [sp.csr_matrix(np.eye(5)) for _ in range(2)]
        test_labels  = [1, 0]
        _write_pkl(train_data,   self._path("train_data.pkl"))
        _write_pkl(train_labels, self._path("train_labels.pkl"))
        _write_pkl(test_data,    self._path("test_data.pkl"))
        _write_pkl(test_labels,  self._path("test_labels.pkl"))

        meta = self._run(
            file_paths={
                "train_data":   self._path("train_data.pkl"),
                "train_labels": self._path("train_labels.pkl"),
                "test_data":    self._path("test_data.pkl"),
                "test_labels":  self._path("test_labels.pkl"),
            },
            agent_name="ehr_agent",
        )
        info = meta.get_metadata_info()
        self.assertEqual(info["status"], "success")
        for key in ("train_data", "train_labels", "test_data", "test_labels"):
            self.assertIn(key, info["dataset"])

    def test_metadata_is_saved_to_disk(self):
        """Metadata.save() is called; the config file should exist on disk."""
        meta = self._run(file_paths={}, agent_name="ehr_agent")
        self.assertTrue(os.path.exists(meta.get_file_path()))

    def test_empty_file_paths_still_succeeds(self):
        meta = self._run(file_paths={}, agent_name="ehr_agent")
        self.assertEqual(meta.get_metadata_info()["status"], "success")

    def test_patient_id_filters_test_data_list(self):
        test_data   = ["note_A", "note_B", "note_C"]
        test_labels = [0, 1, 0]
        _write_pkl(test_data,   self._path("test_data.pkl"))
        _write_pkl(test_labels, self._path("test_labels.pkl"))

        meta = self._run(
            file_paths={"test_data": self._path("test_data.pkl"),
                        "test_labels": self._path("test_labels.pkl")},
            agent_name="note_agent",
            patient_id=1,
        )
        info = meta.get_metadata_info()
        self.assertEqual(info["status"], "success")
        self.assertEqual(info["dataset"]["test_labels"]["configuration"]["num_samples"], 1)

    def test_patient_id_zero_is_accepted(self):
        """patient_id=0 must NOT be treated as falsy."""
        test_data   = ["note_A", "note_B"]
        test_labels = [0, 1]
        _write_pkl(test_data,   self._path("test_data.pkl"))
        _write_pkl(test_labels, self._path("test_labels.pkl"))

        meta = self._run(
            file_paths={"test_data": self._path("test_data.pkl"),
                        "test_labels": self._path("test_labels.pkl")},
            agent_name="note_agent",
            patient_id=0,
        )
        self.assertEqual(meta.get_metadata_info()["status"], "success")

    def test_patient_id_filters_dataframe_test_data(self):
        """When test_data is a DataFrame, step 3 uses .loc[[patient_id]] not list indexing.

        A custom (non-modality) agent_name is used so step 6 (raw_data extraction)
        is skipped — that step assumes list-indexed data for the three known agents.
        """
        import pandas as pd
        df = pd.DataFrame({"a": [10, 20, 30]}, index=[0, 1, 2])
        labels = [0, 1, 0]
        _write_pkl(df,     self._path("test_data.pkl"))
        _write_pkl(labels, self._path("test_labels.pkl"))

        meta = self._run(
            file_paths={"test_data": self._path("test_data.pkl"),
                        "test_labels": self._path("test_labels.pkl")},
            agent_name="custom_agent",  # skips raw_data extraction in step 6
            patient_id=1,
        )
        self.assertEqual(meta.get_metadata_info()["status"], "success")
        self.assertEqual(
            meta.get_metadata_info()["dataset"]["test_labels"]["configuration"]["num_samples"], 1
        )

    def test_csv_is_loaded_as_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_path = self._path("train_data.csv")
        df.to_csv(csv_path, index=False)

        meta = self._run(file_paths={"train_data": csv_path}, agent_name="ehr_agent")
        info = meta.get_metadata_info()
        self.assertEqual(info["status"], "success")
        self.assertEqual(info["dataset"]["train_data"]["configuration"]["num_samples"], 3)

    def test_missing_file_path_produces_unknown_samples(self):
        meta = self._run(
            file_paths={"train_data": "/nonexistent/path/file.pkl"},
            agent_name="ehr_agent",
        )
        info = meta.get_metadata_info()
        self.assertEqual(info["status"], "success")
        self.assertEqual(info["dataset"]["train_data"]["configuration"]["num_samples"], "unknown")

    def test_demographics_not_loaded_without_patient_id(self):
        """Demographics key must be ignored when patient_id is None."""
        demo_list = [{"age": 75.0, "gender": "demographics_GENDER_Female"}]
        _write_pkl(demo_list, self._path("demo.pkl"))

        meta = self._run(
            file_paths={"demographics": self._path("demo.pkl")},
            agent_name="ehr_agent",
            patient_id=None,
        )
        self.assertNotIn("demographic_info", meta.get_metadata_info()["dataset"])

    def test_demographics_saved_when_patient_id_set(self):
        demo_list = [{"age": 75.0, "gender": "demographics_GENDER_Female",
                      "Lab:Memory loss [HIV-SSC] 28389-5": 0,
                      "Lab:APOE gene allele 1 [Identifier] in Blood or Tissue by Molecular genetics method 34731-0": 0}]
        _write_pkl(demo_list, self._path("demo.pkl"))

        meta = self._run(
            file_paths={"demographics": self._path("demo.pkl")},
            agent_name="ehr_agent",
            patient_id=0,
        )
        self.assertIn("demographic_info", meta.get_metadata_info()["dataset"])

    def test_raw_data_saved_for_ehr_agent_with_patient_id(self):
        import scipy.sparse as sp, numpy as np
        test_data = [sp.csr_matrix(np.array([[0.0, 1.0, 0.0]]))]
        _write_pkl(test_data, self._path("test_data.pkl"))

        with patch("cerebra.agents.data_agent.load_ehr_headers", return_value=["f0", "f1", "f2"]):
            meta = self._run(
                file_paths={"test_data": self._path("test_data.pkl")},
                agent_name="ehr_agent",
                patient_id=0,
            )
        self.assertIn("raw_data", meta.get_metadata_info()["dataset"])

    def test_raw_data_saved_for_note_agent_with_patient_id(self):
        test_data = [["note one.\n\nParagraph two.", "note two."]]
        _write_pkl(test_data, self._path("test_data.pkl"))

        meta = self._run(
            file_paths={"test_data": self._path("test_data.pkl")},
            agent_name="note_agent",
            patient_id=0,
        )
        self.assertIn("raw_data", meta.get_metadata_info()["dataset"])

    def test_image_agent_raw_data_is_whole_test_set(self):
        test_data = ["/path/to/scan1.mgz", "/path/to/scan2.mgz"]
        _write_pkl(test_data, self._path("test_data.pkl"))

        meta = self._run(
            file_paths={"test_data": self._path("test_data.pkl")},
            agent_name="image_agent",
            patient_id=0,
        )
        self.assertIn("raw_data", meta.get_metadata_info()["dataset"])


# ---------------------------------------------------------------------------
# 2. Note truncation
# ---------------------------------------------------------------------------
class TestNoteTruncation(unittest.TestCase):

    def setUp(self):
        self.agent = DataAgent()

    def test_short_note_unchanged(self):
        note = "Sentence one. Sentence two."
        data = [[note]]
        result = self.agent._truncate_notes(data, max_sentences=2048)
        self.assertEqual(result[0][0], note)

    def test_long_note_is_trimmed(self):
        note = ". ".join([f"S{i}" for i in range(3000)])
        data = [[note]]
        result = self.agent._truncate_notes(data, max_sentences=10)
        trimmed = result[0][0]
        self.assertLessEqual(len(trimmed.split(".")), 11)  # ≤ 10 sentences + trailing empty

    def test_long_note_keeps_last_sentences(self):
        sentences = [f"Sentence {i}" for i in range(100)]
        note = ". ".join(sentences)
        data = [[note]]
        result = self.agent._truncate_notes(data, max_sentences=5)
        self.assertIn("Sentence 99", result[0][0])
        self.assertNotIn("Sentence 0", result[0][0])

    def test_multiple_patients_and_notes(self):
        long = ". ".join([f"S{i}" for i in range(3000)])
        data = [[long, long], [long]]
        result = self.agent._truncate_notes(data, max_sentences=5)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)


# ---------------------------------------------------------------------------
# 3. _load_data_file
# ---------------------------------------------------------------------------
class TestLoadDataFile(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.agent = DataAgent()

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def test_none_path_returns_none(self):
        self.assertIsNone(self.agent._load_data_file(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.agent._load_data_file(""))

    def test_nonexistent_path_returns_none(self):
        self.assertIsNone(self.agent._load_data_file("/no/such/file.pkl"))

    def test_pkl_round_trip(self):
        obj = {"key": [1, 2, 3]}
        path = self._path("data.pkl")
        _write_pkl(obj, path)
        loaded = self.agent._load_data_file(path)
        self.assertEqual(loaded, obj)

    def test_csv_returns_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({"x": [1, 2]})
        path = self._path("data.csv")
        df.to_csv(path, index=False)
        loaded = self.agent._load_data_file(path)
        self.assertIsInstance(loaded, pd.DataFrame)
        self.assertEqual(len(loaded), 2)


# ---------------------------------------------------------------------------
# 4. get_demographic_info
# ---------------------------------------------------------------------------
class TestGetDemographicInfo(unittest.TestCase):

    def setUp(self):
        self.agent = DataAgent()

    def _demo(self, **kwargs):
        base = {"age": 70.0, "gender": "demographics_GENDER_Male"}
        base.update(kwargs)
        return [base]

    def test_gender_prefix_stripped(self):
        info = self.agent.get_demographic_info(
            [{"age": 80.0, "gender": "demographics_GENDER_Female",
              "Lab:Memory loss [HIV-SSC] 28389-5": 0}], 0)
        self.assertEqual(info["Gender"], "Female")

    def test_gender_plain_string_unchanged(self):
        info = self.agent.get_demographic_info(
            [{"age": 65.0, "gender": "Male"}], 0)
        self.assertEqual(info["Gender"], "Male")

    def test_age_returned(self):
        info = self.agent.get_demographic_info([{"age": 73.5, "gender": "Male"}], 0)
        self.assertAlmostEqual(info["Age"], 73.5)

    def test_apoe_positive_when_any_key_gt_zero(self):
        demo = [{"age": 70.0, "gender": "Male",
                 "Lab:APOE gene allele 1 34731-0": 0,
                 "Lab:APOE gene allele 2 34732-8": 1}]
        self.assertTrue(self.agent.get_demographic_info(demo, 0)["APOE"])

    def test_apoe_negative_when_all_keys_zero(self):
        demo = [{"age": 70.0, "gender": "Male",
                 "Lab:APOE gene allele 1 34731-0": 0}]
        self.assertFalse(self.agent.get_demographic_info(demo, 0)["APOE"])

    def test_apoe_unknown_when_no_apoe_keys(self):
        demo = [{"age": 70.0, "gender": "Male"}]
        self.assertEqual(self.agent.get_demographic_info(demo, 0)["APOE"], "Unknown")

    def test_memory_loss_positive(self):
        demo = [{"age": 70.0, "gender": "Male",
                 "Lab:Memory loss 28389-5": 1}]
        self.assertTrue(self.agent.get_demographic_info(demo, 0)["Memory Loss"])

    def test_memory_loss_negative(self):
        demo = [{"age": 70.0, "gender": "Male",
                 "Lab:Memory impairment 42850-8": 0}]
        self.assertFalse(self.agent.get_demographic_info(demo, 0)["Memory Loss"])

    def test_correct_patient_index_used(self):
        demo = [{"age": 60.0, "gender": "Male"}, {"age": 80.0, "gender": "Female"}]
        info = self.agent.get_demographic_info(demo, 1)
        self.assertAlmostEqual(info["Age"], 80.0)


# ---------------------------------------------------------------------------
# 5. get_ehr_nonzero_features
# ---------------------------------------------------------------------------
class TestGetEhrNonzeroFeatures(unittest.TestCase):

    def setUp(self):
        self.agent = DataAgent()
        self.headers = [f"feature_{i}" for i in range(10)]

    def test_xml_root_tags_present(self):
        import scipy.sparse as sp, numpy as np
        ehr = sp.csr_matrix(np.array([[1.0, 0.0, 2.0]]))
        with patch("cerebra.agents.data_agent.load_ehr_headers", return_value=self.headers):
            xml = self.agent.get_ehr_nonzero_features(ehr)
        self.assertTrue(xml.startswith("<ehr_features>"))
        self.assertTrue(xml.strip().endswith("</ehr_features>"))

    def test_nonzero_values_present(self):
        import scipy.sparse as sp, numpy as np
        ehr = sp.csr_matrix(np.array([[0.0, 1.5, 0.0, 2.0]]))
        with patch("cerebra.agents.data_agent.load_ehr_headers", return_value=self.headers):
            xml = self.agent.get_ehr_nonzero_features(ehr)
        self.assertIn('value="1.5"', xml)
        self.assertIn('value="2.0"', xml)

    def test_zero_values_excluded(self):
        import scipy.sparse as sp, numpy as np
        ehr = sp.csr_matrix(np.array([[0.0, 1.5, 0.0]]))
        with patch("cerebra.agents.data_agent.load_ehr_headers", return_value=self.headers):
            xml = self.agent.get_ehr_nonzero_features(ehr)
        self.assertNotIn('value="0.0"', xml)

    def test_all_zeros_produces_empty_feature_list(self):
        import scipy.sparse as sp, numpy as np
        ehr = sp.csr_matrix(np.zeros((1, 5)))
        with patch("cerebra.agents.data_agent.load_ehr_headers", return_value=self.headers):
            xml = self.agent.get_ehr_nonzero_features(ehr)
        self.assertNotIn("<feature", xml)

    def test_2d_matrix_uses_column_max(self):
        """Multi-row input (multiple visits) should be collapsed via max."""
        import scipy.sparse as sp, numpy as np
        # Row 0: feature 1 = 1.0; Row 1: feature 1 = 3.0 → max = 3.0
        ehr = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 3.0]]))
        with patch("cerebra.agents.data_agent.load_ehr_headers", return_value=self.headers):
            xml = self.agent.get_ehr_nonzero_features(ehr)
        self.assertIn('value="3.0"', xml)
        self.assertNotIn('value="1.0"', xml)

    def test_feature_name_appears_in_xml(self):
        import scipy.sparse as sp, numpy as np
        headers = ["alpha", "beta", "gamma"]
        ehr = sp.csr_matrix(np.array([[0.0, 5.0, 0.0]]))
        with patch("cerebra.agents.data_agent.load_ehr_headers", return_value=headers):
            xml = self.agent.get_ehr_nonzero_features(ehr)
        self.assertIn('name="beta"', xml)

    def test_out_of_range_index_uses_fallback_name(self):
        import scipy.sparse as sp, numpy as np
        ehr = sp.csr_matrix(np.array([[1.0, 2.0, 3.0]]))
        with patch("cerebra.agents.data_agent.load_ehr_headers", return_value=["only_one"]):
            xml = self.agent.get_ehr_nonzero_features(ehr)
        self.assertIn('name="feature_1"', xml)
        self.assertIn('name="feature_2"', xml)


# ---------------------------------------------------------------------------
# 6. get_raw_notes_with_indices
# ---------------------------------------------------------------------------
class TestGetRawNotesWithIndices(unittest.TestCase):

    def setUp(self):
        self.agent = DataAgent()

    def test_total_notes_count(self):
        result = self.agent.get_raw_notes_with_indices(["note A", "note B", "note C"])
        self.assertEqual(result["total_notes"], 3)

    def test_note_index_matches_position(self):
        result = self.agent.get_raw_notes_with_indices(["first", "second"])
        self.assertEqual(result["indexed_notes"][0]["note_index"], 0)
        self.assertEqual(result["indexed_notes"][1]["note_index"], 1)

    def test_double_newline_splits_paragraphs(self):
        result = self.agent.get_raw_notes_with_indices(["Para one.\n\nPara two."])
        self.assertEqual(len(result["indexed_notes"][0]["paragraphs"]), 2)

    def test_no_double_newline_single_paragraph(self):
        result = self.agent.get_raw_notes_with_indices(["Just one paragraph."])
        self.assertEqual(len(result["indexed_notes"][0]["paragraphs"]), 1)

    def test_total_paragraphs_sum(self):
        notes = ["A.\n\nB.", "C."]
        result = self.agent.get_raw_notes_with_indices(notes)
        self.assertEqual(result["total_paragraphs"], 3)

    def test_full_text_preserved(self):
        note = "First.\n\nSecond."
        result = self.agent.get_raw_notes_with_indices([note])
        self.assertEqual(result["indexed_notes"][0]["full_text"], note)

    def test_char_start_and_end_are_integers(self):
        result = self.agent.get_raw_notes_with_indices(["Hello.\n\nWorld."])
        para = result["indexed_notes"][0]["paragraphs"][0]
        self.assertIsInstance(para["char_start"], int)
        self.assertIsInstance(para["char_end"], int)

    def test_empty_paragraphs_skipped(self):
        """Blank lines between paragraphs should not produce empty paragraph entries."""
        result = self.agent.get_raw_notes_with_indices(["Para one.\n\n\n\nPara two."])
        texts = [p["text"] for p in result["indexed_notes"][0]["paragraphs"]]
        self.assertTrue(all(t.strip() for t in texts))


# ---------------------------------------------------------------------------
# 7. _get_feature_descriptions — dynamic agent lookup
# ---------------------------------------------------------------------------
class TestGetFeatureDescriptions(unittest.TestCase):

    def setUp(self):
        self.agent = DataAgent()

    def test_always_includes_label_keys(self):
        for name in ("ehr_agent", "note_agent", "image_agent", None, "unknown_xyz"):
            with self.subTest(agent_name=name):
                with patch("cerebra.agents.data_agent.logger"):  # suppress warning
                    desc = self.agent._get_feature_descriptions(name)
                for label_key in ("train_labels", "validation_labels", "test_labels"):
                    self.assertIn(label_key, desc)

    def test_ehr_agent_descriptions_loaded_from_class(self):
        desc = self.agent._get_feature_descriptions("ehr_agent")
        self.assertIn("train_data", desc)
        self.assertIn("sparse", desc["train_data"].lower())

    def test_note_agent_descriptions_loaded_from_class(self):
        desc = self.agent._get_feature_descriptions("note_agent")
        self.assertIn("train_data", desc)
        self.assertIn("string", desc["train_data"].lower())

    def test_image_agent_descriptions_loaded_from_class(self):
        desc = self.agent._get_feature_descriptions("image_agent")
        self.assertIn("train_data", desc)
        self.assertIn("mri", desc["train_data"].lower())

    def test_unknown_agent_returns_only_label_keys(self):
        with patch("cerebra.agents.data_agent.logger"):
            desc = self.agent._get_feature_descriptions("nonexistent_agent_xyz")
        self.assertNotIn("train_data", desc)
        self.assertIn("train_labels", desc)

    def test_none_agent_name_returns_only_label_keys(self):
        desc = self.agent._get_feature_descriptions(None)
        self.assertNotIn("train_data", desc)
        self.assertIn("train_labels", desc)

    def test_descriptions_are_strings(self):
        desc = self.agent._get_feature_descriptions("ehr_agent")
        for value in desc.values():
            self.assertIsInstance(value, str)


# ---------------------------------------------------------------------------
# 8. data_descriptions on modality agent classes
# ---------------------------------------------------------------------------
class TestModalityAgentDataDescriptions(unittest.TestCase):

    def _import_agent(self, module_name):
        import importlib
        return importlib.import_module(f"cerebra.agents.{module_name}")

    def test_ehr_agent_has_data_descriptions(self):
        mod = self._import_agent("ehr_agent")
        self.assertTrue(hasattr(mod.EHRAgent, "data_descriptions"))

    def test_note_agent_has_data_descriptions(self):
        mod = self._import_agent("note_agent")
        self.assertTrue(hasattr(mod.NoteAgent, "data_descriptions"))

    def test_image_agent_has_data_descriptions(self):
        mod = self._import_agent("image_agent")
        self.assertTrue(hasattr(mod.ImageAgent, "data_descriptions"))

    def test_ehr_agent_descriptions_have_required_splits(self):
        from cerebra.agents.ehr_agent import EHRAgent
        for key in ("train_data", "validation_data", "test_data"):
            self.assertIn(key, EHRAgent.data_descriptions)

    def test_note_agent_descriptions_have_required_splits(self):
        from cerebra.agents.note_agent import NoteAgent
        for key in ("train_data", "validation_data", "test_data"):
            self.assertIn(key, NoteAgent.data_descriptions)

    def test_image_agent_descriptions_have_required_splits(self):
        from cerebra.agents.image_agent import ImageAgent
        for key in ("train_data", "validation_data", "test_data"):
            self.assertIn(key, ImageAgent.data_descriptions)

    def test_descriptions_are_all_strings(self):
        from cerebra.agents.ehr_agent import EHRAgent
        from cerebra.agents.note_agent import NoteAgent
        from cerebra.agents.image_agent import ImageAgent
        for cls in (EHRAgent, NoteAgent, ImageAgent):
            for v in cls.data_descriptions.values():
                self.assertIsInstance(v, str, f"{cls.__name__}.data_descriptions value must be str")


# ---------------------------------------------------------------------------
# 9. _wrap_sql_result
# ---------------------------------------------------------------------------
class TestWrapSqlResult(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        patcher = patch("cerebra.agents.data_agent.CACHE_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.agent = DataAgent()

    def test_returns_metadata(self):
        from cerebra.utils.metadata import Metadata
        result = {"status": "success", "cohort_size": 100, "split": {"train": []}}
        meta = self.agent._wrap_sql_result(result, "training")
        self.assertIsInstance(meta, Metadata)

    def test_status_success_propagated(self):
        result = {"status": "success", "data": [1, 2, 3]}
        meta = self.agent._wrap_sql_result(result, "training")
        self.assertEqual(meta.get_metadata_info()["status"], "success")

    def test_status_error_propagated(self):
        result = {"status": "error", "message": "something failed"}
        meta = self.agent._wrap_sql_result(result, "inference")
        self.assertEqual(meta.get_metadata_info()["status"], "error")

    def test_non_status_keys_appear_in_dataset(self):
        result = {"status": "success", "cohort_size": 42, "patient_ids": ["a", "b"]}
        meta = self.agent._wrap_sql_result(result, "training")
        dataset = meta.get_metadata_info()["dataset"]
        self.assertIn("cohort_size", dataset)
        self.assertIn("patient_ids", dataset)
        self.assertNotIn("status", dataset)

    def test_dataset_entries_have_saved_path(self):
        result = {"status": "success", "timeline": [{"event": "lab"}]}
        meta = self.agent._wrap_sql_result(result, "inference")
        entry = meta.get_metadata_info()["dataset"]["timeline"]
        self.assertIn("saved_path", entry)
        self.assertTrue(os.path.exists(entry["saved_path"]))

    def test_dataset_entries_have_num_samples(self):
        result = {"status": "success", "patient_ids": ["p1", "p2", "p3"]}
        meta = self.agent._wrap_sql_result(result, "training")
        entry = meta.get_metadata_info()["dataset"]["patient_ids"]
        self.assertEqual(entry["configuration"]["num_samples"], 3)

    def test_empty_result_dict(self):
        result = {"status": "success"}
        meta = self.agent._wrap_sql_result(result, "exploration")
        self.assertEqual(meta.get_metadata_info()["dataset"], {})

    def test_metadata_is_persisted_to_disk(self):
        result = {"status": "success", "x": [1]}
        meta = self.agent._wrap_sql_result(result, "training")
        self.assertTrue(os.path.exists(meta.get_file_path()))


if __name__ == "__main__":
    unittest.main()
