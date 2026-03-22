from cerebra.agents.lightweight_agent import LightweightAgent
from typing import Dict, Any
from cerebra.utils.metadata import Metadata
import os

class EHRAgent(LightweightAgent):
    data_descriptions = {
        "train_data":      "EHR data for training, list of sparse matrices",
        "validation_data": "EHR data for validation, list of sparse matrices",
        "test_data":       "EHR data for testing, list of sparse matrices",
    }

    def __init__(self, llm_engine_name: str = "gpt-4o"):
        super().__init__(
            agent_name="ehr_agent",
            llm_engine_name=llm_engine_name,
            enabled_tools=["ehr_model_trainer", "ehr_model_inference"],
            verbose=True
        )

    def run(self, task: str, input_metadata: Metadata) -> Metadata:
        """Extract features from EHR data and return as Metadata"""

        # Process using existing reasoning
        result = self.reason_and_execute(task, input_metadata)
        status = result.get_metadata_info()["status"]

        if status == "success":
            # agent_output contains full metadata info from tool execution
            agent_output = result.get_metadata_info()["dataset"]['agent_output']
            # Extract the dataset from the tool's metadata
            output_data = agent_output.get('dataset', {})
        else:
            output_data = {}
        return output_data

    def register_agent_capabilities(self) -> Dict[str, Any]:
        return {
            "ehr_agent": {
                "agent_description": "EHR agent for training models (using ehr_model_trainer tool) and making prediction (using ehr_model_inference tool), providing evidences for each prediction on EHR data",
                "agent_capabilities": [
                    "training models on EHR data",
                    "performing prediction on the test data, providing prediction results, prediction confidence and corresponding evidences for each prediction"
                ],
                "agent_input_types": {"task": "str", "input_metadata": "Metadata"},
                "agent_output_type": "Metadata object containing saved paths to features extracted from the EHR data, model and prediction results"
            }
        }

