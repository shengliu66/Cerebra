# agents/data_agent.py
from typing import Dict, Any, List, Optional
from cerebra.utils.metadata import Metadata
from cerebra.utils.utils import save_to_pickle
import os
import numpy as np
import pickle
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import re
from cerebra.utils.ehr_headers import load_ehr_headers
load_dotenv(find_dotenv())
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")


class DataAgent:
    def __init__(self):
        self.root_data_path = '/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/20251207/rolling_not_long_island/'
        self.root_data_path_long_island = '/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/20251207/rolling_long_island/'
        self.metadata = None

    # ==================== Helper methods for run() ====================
    
    def _load_data_file(self, path: str):
        """Load data file (pickle or CSV) if it exists."""
        if not os.path.exists(path):
            return None
        
        if path.endswith('.csv'):
            import pandas as pd
            return pd.read_csv(path)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    def _get_base_data_path(self, institution: str, time_to_event: bool, diagnosis: bool, year: int) -> str:
        """Determine the base data directory based on parameters."""
        if institution == "NYU":
            root = self.root_data_path
            diagnosis_path = '/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/diagnosis_not_long_island/'
            time_to_event_path =  '/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/mri_indexed_not_long_island/time_to_event/'
            #'/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/20251207/mri_indexed_not_long_island/time_to_event/'
            #'/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/mri_indexed_not_long_island/time_to_event/'
        elif institution == "LongIsland":
            root = self.root_data_path_long_island
            diagnosis_path = '/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/diagnosis_long_island/'
            time_to_event_path = '/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/mri_indexed_long_island/time_to_event/'
            #'/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/20251207/mri_indexed_long_island/time_to_event/'
            #'/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/mri_indexed_long_island/time_to_event/'
        else:
            raise ValueError(f"Invalid institution: {institution}")
        
        if time_to_event:
            return time_to_event_path
        elif diagnosis:
            return diagnosis_path
        else:
            return f'{root}/180days_blackout_{year}yr_label/'
    
    def _get_data_file_info(self, agent_name: str, volume: bool = False) -> tuple:
        """Get file prefix and extension for the given agent."""
        if agent_name == "ehr_agent":
            return "X_ehr", "_4447_header_phecode.pkl"
        elif agent_name == "note_agent":
            return "X_note", ".pkl"
        elif agent_name == "image_agent":
            if volume:
                return "X_mri_volume", ".csv"
            return "X_mri", ".pkl"
        else:
            raise ValueError(f"Invalid agent name: {agent_name}")
    def _replace_rolling_long_island(self, text: str) -> str:
        result = re.sub(
        r'(?<!rolling_not_)rolling_long_island',
        'rolling_not_long_island',
        text
        )
        result = re.sub(
        r'(?<!mri_indexed_not_)mri_indexed_long_island',
        'mri_indexed_not_long_island',
        result
        )
        result = re.sub(
        r'(?<!diagnosis_not_)diagnosis_long_island',
        'diagnosis_not_long_island',
        result
        )
        return result
    def _load_train_val_test_data(self, base_path: str, agent_name: str, volume: bool = False) -> dict:
        """Load train/val/test data for a given agent."""
        prefix, ext = self._get_data_file_info(agent_name, volume)
        return {
            "train": self._load_data_file(f"{self._replace_rolling_long_island(base_path)}{prefix}_train{ext}"),
            "val": self._load_data_file(f"{self._replace_rolling_long_island(base_path)}{prefix}_val{ext}"),
            "test": self._load_data_file(f"{base_path}{prefix}_test{ext}"),
        }
    
    def _load_labels(self, base_path: str, time_to_event: bool) -> dict:
        """Load labels, handling both regular and time-to-event cases."""
        if not time_to_event:
            return {
                "train": self._load_data_file(f"{self._replace_rolling_long_island(base_path)}Y_train.pkl"),
                "val": self._load_data_file(f"{self._replace_rolling_long_island(base_path)}Y_val.pkl"),
                "test": self._load_data_file(f"{base_path}Y_test.pkl"),
            }
        
        # Time-to-event: load times and indicators separately, then combine
        times = {
            "train": self._load_data_file(f"{self._replace_rolling_long_island(base_path)}Y_time_to_event_train.pkl"),
            "val": self._load_data_file(f"{self._replace_rolling_long_island(base_path)}Y_time_to_event_val.pkl"),
            "test": self._load_data_file(f"{base_path}Y_time_to_event_test.pkl"),
        }
        
        indicators = {
            "train": self._load_data_file(f"{self._replace_rolling_long_island(base_path)}Y_train.pkl"),
            "val": self._load_data_file(f"{self._replace_rolling_long_island(base_path)}Y_val.pkl"),
            "test": self._load_data_file(f"{base_path}Y_test.pkl"),
        }

        


        # Validate and combine into (time, indicator) tuples
        labels = {}
        for split in ["train", "val", "test"]:
            assert len(times[split]) == len(indicators[split]), \
                f"Mismatch in {split}: times={len(times[split])}, indicators={len(indicators[split])}"
            labels[split] = list(zip(times[split], indicators[split]))
        
        return labels
    
    def _validate_data_label_lengths(self, data: dict, labels: dict) -> None:
        """Ensure data and labels have matching lengths for each split."""
        for split in ["train", "val", "test"]:
            if data[split] is None or labels[split] is None:
                continue
            data_len = len(data[split])
            label_len = len(labels[split])
            if data_len != label_len:
                raise ValueError(
                    f"Mismatch in {split}: data has {data_len} samples, "
                    f"labels have {label_len} samples"
                )
    
    def _truncate_notes(self, data: list, max_sentences: int = 2048) -> list:
        """Truncate notes to keep only the last max_sentences sentences."""
        for i in range(len(data)):
            patient_data = data[i]  # a list of texts
            for j, text in enumerate(patient_data):
                sentences = text.replace('..', '').split('.')
                if len(sentences) > max_sentences:
                    last_sentences = sentences[-max_sentences:]
                    data[i][j] = '.'.join(last_sentences)
        return data
    
    def _get_feature_descriptions(self, agent_name: str) -> dict:
        """Get feature descriptions for the given agent."""
        descriptions_map = {
            "ehr_agent": {
                "train_data": "EHR data for training, list of sparse matrices",
                "validation_data": "EHR data for validation, list of sparse matrices",
                "test_data": "EHR data for testing, list of sparse matrices",
            },
            "note_agent": {
                "train_data": "Note data for training, list of strings",
                "validation_data": "Note data for validation, list of strings",
                "test_data": "Note data for testing, list of strings",
            },
            "image_agent": {
                "train_data": "MRI data for training, list of path to the MRI files",
                "validation_data": "MRI data for validation, list of path to the MRI files",
                "test_data": "MRI data for testing, list of path to the MRI files",
            },
        }
        
        base = descriptions_map.get(agent_name, {})
        # Add common label descriptions
        base.update({
            "train_labels": "Labels for training, list of labels",
            "validation_labels": "Labels for validation, list of labels",
            "test_labels": "Labels for testing, list of labels",
        })
        return base

    def run(self, task: str, institution: str = "NYU", agent_name: str = None, 
            patient_id: str = None, year: int = 1, time_to_event: bool = False, 
            diagnosis: bool = False, volume: bool = False) -> Metadata:
        """Load and process data, return as Metadata."""
        
        # 1. Get base data path
        base_path = self._get_base_data_path(institution, time_to_event, diagnosis, year)
        
        # 2. Load data and labels
        data = self._load_train_val_test_data(base_path, agent_name, volume)
        labels = self._load_labels(base_path, time_to_event)

        
        # 3. Validate data and labels have matching lengths
        self._validate_data_label_lengths(data, labels)
        
        print(f"# Training samples: {len(labels['train'])}")
        print(f"# Validation samples: {len(labels['val'])}")
        print(f"# Testing samples: {len(labels['test'])}")
        
        # 4. Apply agent-specific transformations
        if agent_name == "note_agent":
            data["train"] = self._truncate_notes(data["train"])
        
        # 5. Build filtered data (filter for specific patient if needed)
        
        if isinstance(data["test"], pd.DataFrame):
            test_data_value = data["test"].loc[[int(patient_id)]] if patient_id is not None else data["test"]
        else:
            test_data_value = [data["test"][int(patient_id)]] if patient_id is not None else data["test"]
    
        
        filtered_data = {
            "train_data": data["train"],
            "validation_data": data["val"],
            "test_data": test_data_value,
            "train_labels": labels["train"],
            "validation_labels": labels["val"],
            "test_labels": [labels["test"][int(patient_id)]] if patient_id is not None else labels["test"],
        }


        # 6. Get raw data for patient if specified (for downstream use)
        demographic_info = None
        raw_data = None
        if patient_id is not None:
            demographic_info = self.get_demographic_info(int(patient_id), year=year)
            if agent_name == "ehr_agent":
                raw_data = self.get_ehr_nonzero_features(data["test"][int(patient_id)])
            elif agent_name == "note_agent":
                raw_data = self.get_raw_notes_with_indices(data["test"][int(patient_id)])
            elif agent_name == "image_agent":
                raw_data = data["test"]
        
        # 7. Save data and create metadata
        feature_descriptions = self._get_feature_descriptions(agent_name)
        
        cache_dir = os.path.join(CACHE_DIR, "data_agent", agent_name, f'{institution}_{year}yr_survival_{time_to_event}_diagnosis_{diagnosis}_volume_{volume}')
        os.makedirs(cache_dir, exist_ok=True)
        
        saved_paths = {}
        for key, value in filtered_data.items():
            if 'train' in key or 'val' in key:
                check_exist = True
                key_with_prefix = key
            else:
                check_exist = False
                key_with_prefix = f"{institution}_{patient_id}_{key}" if patient_id is not None else key
            save_path = save_to_pickle(value, key_with_prefix, cache_dir, check_exist=check_exist)
            saved_paths[key] = save_path
        
        # Save demographic info and raw data for downstream use if patient_id is specified
        if patient_id is not None and demographic_info is not None:
            demo_save_path = save_to_pickle(demographic_info, f"{institution}_{patient_id}_demographic_info", cache_dir, check_exist=False)
            saved_paths['demographic_info'] = demo_save_path
        
        if patient_id is not None and raw_data is not None:
            raw_data_save_path = save_to_pickle(raw_data, f"{institution}_{patient_id}_raw_data", cache_dir, check_exist=False)
            saved_paths['raw_data'] = raw_data_save_path
        
        output_dataset = {
            k: {
                "saved_path": saved_paths[k],
                "description": feature_descriptions.get(k, ""),
                "configuration": {
                    "num_samples": len(filtered_data[k]) if hasattr(filtered_data[k], '__len__') else "unknown"
                }
            } for k in filtered_data
        }
        
        # Add demographic info and raw data to output_dataset if they were saved
        if patient_id is not None and demographic_info is not None:
            output_dataset['demographic_info'] = {
                "saved_path": saved_paths['demographic_info'],
                "description": "Patient demographic information",
                "configuration": {}
            }
        
        if patient_id is not None and raw_data is not None:
            output_dataset['raw_data'] = {
                "saved_path": saved_paths['raw_data'],
                "description": f"Raw {agent_name} data for patient",
                "configuration": {}
            }
        
        cache_directory = os.path.join(CACHE_DIR, "data_agent")
        self.metadata = Metadata.create_agent_output(
            status="success",
            dataset=output_dataset,
            model={},
            cache_directory=cache_directory,
            agent_name="data_agent"
        )
        
        self.metadata.save()
        print(f"metadata config saved to {self.metadata.metadata_config_path}")
        return self.metadata
    
    def get_demographic_info(self, patient_id: int, year: int = 1) -> Dict[str, Any]:
        """Get demographic information for a patient"""
        data_path = f'{self.root_data_path}/180days_blackout_{year}yr_label/demographics_info_test.pkl'
        with open(data_path, 'rb') as f:
            demographic_info = pickle.load(f)
        demographic_info = demographic_info[patient_id]
        
        # Extract APOE information
        apoe_keys = [k for k in demographic_info.keys() if 'APOE' in k]
        apoe_values = [demographic_info[k] for k in apoe_keys]
        if any(v > 0 for v in apoe_values):
            apoe_status = True
        elif any(v == 0 for v in apoe_values):
            apoe_status = False
        else:
            apoe_status = "Unknown"
        
        # Extract Memory Loss information
        memory_keys = [k for k in demographic_info.keys() if 'Memory' in k or 'memory' in k]
        memory_values = [demographic_info[k] for k in memory_keys]
        if any(v > 0 for v in memory_values):
            memory_loss = True
        elif any(v == 0 for v in memory_values):
            memory_loss = False
        else:
            memory_loss = "Unknown"
        
        # Extract gender (remove prefix if present)
        gender = demographic_info.get('gender', 'Unknown')
        if isinstance(gender, str) and 'demographics_GENDER_' in gender:
            gender = gender.replace('demographics_GENDER_', '')
        
        return {
            'APOE': apoe_status,
            'Age': demographic_info.get('age', 'Unknown'),
            'Gender': gender,
            'Memory Loss': memory_loss
        }
    
    def get_raw_notes_with_indices(self, note_data: List[str]) -> Dict[str, Any]:
        """
        Get raw notes with paragraph/sentence indices for traceability.
        
        Args:
            note_data: List of note strings (e.g., from data_test[patient_id])
            
        Returns:
            Dict with indexed notes structure for chat retrieval
        """
        indexed_notes = []
        
        for note_idx, note in enumerate(note_data):
            # Split into paragraphs (double newline) or keep as single unit
            paragraphs = note.split('\n\n') if '\n\n' in note else [note]
            
            note_entry = {
                'note_index': note_idx,
                'full_text': note,
                'paragraphs': []
            }
            
            for para_idx, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    note_entry['paragraphs'].append({
                        'paragraph_index': para_idx,
                        'text': paragraph.strip(),
                        'char_start': note.find(paragraph),
                        'char_end': note.find(paragraph) + len(paragraph)
                    })
            
            indexed_notes.append(note_entry)
        
        return {
            'indexed_notes': indexed_notes,
            'total_notes': len(note_data),
            'total_paragraphs': sum(len(n['paragraphs']) for n in indexed_notes)
        }
    
    def get_ehr_nonzero_features(self, ehr_data) -> str:
        """Get non-zero features for a patient in XML format"""
        ehr_header_names = load_ehr_headers()
        
        # Take max across rows if 2D, otherwise keep as is
        if len(ehr_data.shape) > 1:
            patient_vector = ehr_data.max(axis=0).toarray().flatten()
        else:
            patient_vector = ehr_data.toarray().flatten() if hasattr(ehr_data, 'toarray') else ehr_data
        
        # Get non-zero indices and values
        nonzero_indices = np.nonzero(patient_vector)[0]
        
        # Build XML string
        xml_parts = ['<ehr_features>']
        for idx in nonzero_indices:
            feature_name = ehr_header_names[idx] if idx < len(ehr_header_names) else f"feature_{idx}"
            value = patient_vector[idx]
            xml_parts.append(f'  <feature name="{feature_name}" value="{value}"/>')
        xml_parts.append('</ehr_features>')
        
        return '\n'.join(xml_parts) 
    

    def get_test_data(self, institution: str = "NYU", agent_name: str = None, year: int = 1, time_to_event: bool = False) -> tuple:
        """Load and process test data, return as tuple of data and labels."""
        # Determine base path (uses legacy hardcoded paths for backward compatibility)
        legacy_paths = {
            "NYU": {
                "time_to_event": '/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/rolling_not_long_island/time_to_event/',
                "default": f'/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/rolling_not_long_island/{year}yr_blackout_2yr_label/'
            },
            "LongIsland": {
                "time_to_event": '/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/rolling_long_island/time_to_event/',
                "default": f'/gpfs/data/razavianlab/longchen/agentic/data/nyu_dataset/rolling_long_island/{year}yr_blackout_2yr_label/'
            }
        }
        
        if institution not in legacy_paths:
            raise ValueError(f"Invalid institution: {institution}")
        
        base_path = legacy_paths[institution]["time_to_event" if time_to_event else "default"]
        
        # Load test data using helper
        prefix, ext = self._get_data_file_info(agent_name, volume=False)
        data_test = self._load_data_file(f"{base_path}{prefix}_test{ext}")
        
        # Load labels
        if not time_to_event:
            labels_test = self._load_data_file(f"{base_path}Y_test.pkl")
        else:
            event_times_test = self._load_data_file(f"{base_path}Y_test_time_to_label.pkl")
            event_indicators_test = self._load_data_file(f"{base_path}Y_test.pkl")
            assert len(event_times_test) == len(event_indicators_test)
            labels_test = list(zip(event_times_test, event_indicators_test))
        
        return data_test, labels_test