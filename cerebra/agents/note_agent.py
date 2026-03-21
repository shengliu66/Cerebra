from cerebra.agents.lightweight_agent import LightweightAgent
from typing import Dict, Any
from cerebra.utils.metadata import Metadata  # ✅ Use Metadata only
import os
import uuid

# cerebra/agents/note_agent.py
class NoteAgent(LightweightAgent):
    data_descriptions = {
        "train_data":      "Note data for training, list of strings",
        "validation_data": "Note data for validation, list of strings",
        "test_data":       "Note data for testing, list of strings",
    }

    def __init__(self, llm_engine_name: str = "gpt-4o"):
        super().__init__(
            agent_name="note_agent",
            llm_engine_name=llm_engine_name,
            enabled_tools=["note_model_trainer", "note_model_inference"],
            verbose=True
        )

    def run(self, task: str, input_metadata: Metadata) -> Metadata:
        """
        Extract features from EHR data and return as Metadata
        """
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
            "note_agent": {
                "agent_description": "Note agent for training models and making prediction, providing evidences for each prediction on note data",
                "agent_capabilities": [
                    "training models on note data",
                    "performing prediction on the test data, providing prediction results, prediction confidence and corresponding evidences for each prediction"
                ],
                "agent_input_types": {"task": "str", "input_metadata": "Metadata"},
                "agent_output_type": "Metadata object containing saved paths to features extracted from the note data, model and prediction results"
            }
        }