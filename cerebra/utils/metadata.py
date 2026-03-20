import os
import pickle
from typing import Dict, Any, Literal, Optional, List

class Metadata:
    def __init__(self, content: Dict[str, Any], cache_dir: str, agent_name: str):
        self.cache_dir = cache_dir  # Directory for agent run
        self.agent_name = agent_name
        self.agent_dir = os.path.join(cache_dir, agent_name)
        self.dataset_dir = os.path.join(self.agent_dir, "dataset")
        self.model_dir = os.path.join(self.agent_dir, "model")
        self.metadata_config_path = os.path.join(self.agent_dir, "metadata.config")

        os.makedirs(self.dataset_dir, exist_ok=True)  # Ensure dataset directory exists
        os.makedirs(self.model_dir, exist_ok=True)  # Ensure model directory exists

        self.__metadata = content
        self.save()  # Always persist metadata

    def save(self):
        """Save the metadata object to a file."""
        with open(self.metadata_config_path, 'wb') as f:
            pickle.dump(self.__metadata, f)

    def get_metadata_info(self) -> Dict[str, Any]:
        """Retrieve metadata content."""
        return self.__metadata

    def get_file_path(self) -> str:
        """Return the path to the metadata config file."""
        return self.metadata_config_path

    def update_metadata(self,
                        section: Literal['dataset', 'model'],
                        name: str,
                        saved_path: str,
                        description: str,
                        configuration: Dict[str, Any]) -> None:
        """Update metadata with dataset or model information."""
        entry = {
            "saved_path": saved_path,
            "description": description,
            "configuration": configuration
        }
        self.__metadata[section][name] = entry
        self.save()

    def merge_metadata(self, other: 'Metadata') -> None:
        """Merge metadata from another Metadata object."""
        other_content = other.get_metadata_info()
        for section in ["dataset", "model"]:
            for name, entry in other_content.get(section, {}).items():
                self.update_metadata(
                    section=section,
                    name=name,
                    saved_path=entry["saved_path"] if "saved_path" in entry else None,
                    description=entry.get("description", ""),
                    configuration=entry.get("configuration", {})
                )

    @classmethod
    def create_agent_output(
        cls,
        status: str,
        dataset: Dict[str, Dict[str, Any]] = {},
        model: Dict[str, Dict[str, Any]] = {},
        cache_directory: str = "cerebra_cache",
        agent_name: str = "note_agent",
        status_description: str = "Pipeline execution status"
    ) -> 'Metadata':
        """Create a Metadata object for agent output."""
        # Here, hash_folder no longer depends on uuid, you can use a constant folder name or
        # a timestamp or something else depending on your needs.
        content = {
            "status": status,
            "dataset": dataset,
            "model": model,
            "final": None
        }
        return cls(content, cache_directory, agent_name)

    @classmethod
    def create_empty_metadata(cls, cache_directory: str = "cerebra_cache", agent_name: str = "note_agent") -> 'Metadata':
        """Create a Metadata object for agent output."""
        content = {
            "status": "success",
            "dataset": {},
            "model": {},
            "final": None
        }
        return cls(content, cache_directory, agent_name)
    
    @staticmethod
    def load_all_agent_runs(agent_name: str, cache_directory: str = "cerebra_cache") -> List['Metadata']:
        """Load all metadata for agent runs from the specified cache directory."""
        results = []
        for agent_run_folder in os.listdir(cache_directory):
            full_hash_path = os.path.join(cache_directory, agent_run_folder)
            agent_path = os.path.join(full_hash_path, agent_name, "metadata.config")
            if os.path.exists(agent_path):
                try:
                    with open(agent_path, 'rb') as f:
                        content = pickle.load(f)
                    results.append(Metadata(content, full_hash_path, agent_name))
                except Exception:
                    continue  # skip broken files
        return results

    