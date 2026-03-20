from typing import Dict, Any, Literal
import os
import pickle
import uuid

class Dataset:
    def __init__(
        self,
        dataset: Dict[str, Any],
        dataset_type: Literal['raw_dataset', 'agent_output'],
        dataset_description: str,
        feature_description: Dict[str, str],
        cache_directory: str,
        ):

        assert set(dataset.keys()) == set(feature_description.keys())  # Make sure data completeness

        # Generate a unique hash for the file name
        file_path = os.path.join(cache_directory, f"{uuid.uuid4().hex}.pkl")

        # Save the data to the pickle file
        os.makedirs(cache_directory, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        self.__dataset_file_path = file_path

        self.__dataset_type = dataset_type
        self.__dataset_description = dataset_description
        self.__feature_description = feature_description
        self.__dataset_len = len(dataset.get(list(dataset.keys())[0], []))

    def get_dataset(
        self,
        ):
        """
            Get all attributes of the dataset as Python instances

            Return:
                dataset (Dict[str, Any]): Data where keys are the features and values are a list of values
                dataset_type (Literal['raw_dataset', 'agent_output', 'tool_output']): Indicates type of this dataset
                dataset_description (str) : Description of this dataset
                feature_description (Dict[str, str]): Description of each feature
        """
        with open(self.__dataset_file_path, 'rb') as f:
            dataset = pickle.load(f)

        return {
            "dataset": dataset,
            "dataset_type": self.__dataset_type,
            "dataset_description": self.__dataset_description,
            "feature_description": self.__feature_description,
            "dataset_len": self.__dataset_len,
        }

    def __len__(self):
        return self.__dataset_len

    def get_dataset_info(
            self,
            ) -> str:
        """
            Get a string that describes the dataset and feature

            Return:
                dataset_info_str (str)
        """


        return {
            "dataset_type": self.__dataset_type,
            "dataset_description": self.__dataset_description,
            "feature_description": self.__feature_description,
            "dataset_len": self.__dataset_len,
        }

    @classmethod
    def create_agent_output(cls,
                           processed_data: Dict[str, Any],
                           description: str,
                           feature_descriptions: Dict[str, str],
                           cache_directory: str) -> 'Dataset':
        """Factory method to create Dataset from agent output"""
        return cls(
            dataset=processed_data,
            dataset_type='agent_output',
            dataset_description=description,
            feature_description=feature_descriptions,
            cache_directory=cache_directory
        )

    def update_dataset(self, other_dataset: 'Dataset'):
        """
        Merge this dataset with another dataset, intelligently handling different data types.

        This method creates a new dataset by combining the current dataset with another dataset.
        It handles various data types appropriately:
        - Lists: Extends existing lists with new values
        - Dictionaries: Recursively merges dictionaries
        - Other types: Overwrites with new values (with conflict resolution)

        Args:
            other_dataset (Dataset): The dataset to merge with this one

        Returns:
            Dataset: A new dataset containing the merged data

        Raises:
            TypeError: If other_dataset is not a Dataset instance
            ValueError: If datasets have incompatible structures
        """
        if not isinstance(other_dataset, Dataset):
            raise TypeError(f"Expected Dataset instance, got {type(other_dataset)}")

        # Get current dataset data
        current_data = self.get_dataset()["dataset"]
        current_features = self.get_dataset()["feature_description"]

        # Get other dataset data
        other_data = other_dataset.get_dataset()["dataset"]
        other_features = other_dataset.get_dataset()["feature_description"]

        # Initialize merged data with deep copy of current data
        merged_data = self._deep_copy_data(current_data)

        # Merge data from other dataset
        for key, values in other_data.items():
            if key in merged_data:
                merged_data[key] = self._merge_values(merged_data[key], values, key)
            else:
                # Add new feature
                merged_data[key] = self._deep_copy_data(values)

        # Create merged feature descriptions
        merged_feature_descriptions = {}
        merged_feature_descriptions.update(current_features)
        merged_feature_descriptions.update(other_features)

        # Create descriptive merged description
        current_desc = self.get_dataset()['dataset_description']
        other_desc = other_dataset.get_dataset()['dataset_description']
        merged_description = f"Merged dataset: [{current_desc}] + [{other_desc}]"

        # Determine appropriate dataset type
        merged_type = self._determine_merged_type(
            self.__dataset_type,
            other_dataset.__dataset_type
        )

        return Dataset(
            dataset=merged_data,
            dataset_type=merged_type,
            dataset_description=merged_description,
            feature_description=merged_feature_descriptions,
            cache_directory=os.path.dirname(self.__dataset_file_path)
        )

    def _deep_copy_data(self, data):
        """
        Create a deep copy of data, handling different data types appropriately.

        Args:
            data: The data to copy

        Returns:
            Deep copy of the data
        """
        if isinstance(data, list):
            return [self._deep_copy_data(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._deep_copy_data(v) for k, v in data.items()}
        else:
            return data

    def _merge_values(self, current_value, new_value, key):
        """
        Merge two values intelligently based on their types.

        Args:
            current_value: Existing value in the dataset
            new_value: New value to merge
            key: The key name for context in error messages

        Returns:
            Merged value
        """
        # Handle list merging
        if isinstance(current_value, list) and isinstance(new_value, list):
            merged_list = current_value.copy()
            merged_list.extend(new_value)
            return merged_list

        # Handle dictionary merging
        elif isinstance(current_value, dict) and isinstance(new_value, dict):
            merged_dict = current_value.copy()
            for k, v in new_value.items():
                if k in merged_dict:
                    merged_dict[k] = self._merge_values(merged_dict[k], v, f"{key}.{k}")
                else:
                    merged_dict[k] = self._deep_copy_data(v)
            return merged_dict

        # Handle type conflicts - prioritize new value but warn about conflicts
        elif type(current_value) != type(new_value):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Type conflict for key '{key}': {type(current_value)} vs {type(new_value)}. Using new value.")
            return new_value

        # Handle same types that are not lists or dicts
        else:
            # For same types, prefer new value (this could be made configurable)
            return new_value

    def _determine_merged_type(self, type1, type2):
        """
        Determine the appropriate dataset type for merged data.

        Args:
            type1: First dataset type
            type2: Second dataset type

        Returns:
            Appropriate merged dataset type
        """
        # If both are the same type, keep it
        if type1 == type2:
            return type1

        # If one is agent_output, prefer that
        if type1 == 'agent_output' or type2 == 'agent_output':
            return 'agent_output'

        # Default to agent_output for mixed types
        return 'agent_output'
