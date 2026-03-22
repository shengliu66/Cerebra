from cerebra.agents.lightweight_agent import LightweightAgent
from typing import Dict, Any
from cerebra.utils.metadata import Metadata  # ✅ Use Metadata only
import os
import uuid

# cerebra/agents/image_agent.py
class ImageAgent(LightweightAgent):
    data_descriptions = {
        "train_data":      "MRI data for training, list of paths to MRI files",
        "validation_data": "MRI data for validation, list of paths to MRI files",
        "test_data":       "MRI data for testing, list of paths to MRI files",
    }

    def __init__(self, llm_engine_name: str = "gpt-4o"):
        super().__init__(
            agent_name="image_agent",
            llm_engine_name=llm_engine_name,
            enabled_tools=["image_model_trainer", "image_model_inference"],
            verbose=True
        )

    def run(self, task: str, input_metadata: Metadata) -> Metadata:
        """
        Analyze images for disease detection or other purposes and return as Metadata
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
            "image_agent": {
                "agent_description": "Image agent for training models and making prediction, providing evidences for each prediction on image data",
                "agent_capabilities": [
                    "training models on image data",
                    "performing prediction on the test data, providing prediction results, prediction confidence and corresponding evidences for each prediction"
                ],
                "agent_input_types": {"task": "str", "input_metadata": "Metadata"},
                "agent_output_type": "Metadata object containing saved paths to features extracted from the image data, model and prediction results"
            }
        }