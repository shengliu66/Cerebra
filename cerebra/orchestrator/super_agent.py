# cerebra/orchestrator/super_agent.py
from cerebra.orchestrator.base import BaseOrchestrator
from cerebra.utils.metadata import Metadata
from typing import Dict, Any, Union

class SuperAgent(BaseOrchestrator):
    def __init__(self, llm_engine_name: str = "gpt-4o", **kwargs):
        super().__init__(
            llm_engine_name=llm_engine_name,
            orchestrator_name="super_agent",
            enabled_agents=[
                # "data_agent",  # Always load data first
                "ehr_agent",
                "note_agent",
                "image_agent",
                "summary_agent",
            ],
            max_steps=15,
            verbose=True,
            **kwargs
        )

    def run(self, task: str, patient_id: str, year: int = 1, institution: str = "NYU", diagnosis: bool = False, time_to_event: bool = False, volume: bool = False) -> Metadata:
        """
        Run the super agent with internal data loading

        Args:
            task: The main task to accomplish (e.g., "Predict dementia risk for patient X")
            patient_id: The ID of the patient to predict the dementia risk for
            year: The number of years to predict the dementia risk for
            context: Additional context

        Returns:
            Dataset: Complete orchestration result as a Dataset object
        """
        # Step 1: Execute the main orchestration task
        system_prompt = """
You are a super agent that accomplishes a query from the user.
This is the query from the user: {task}

You can accomplish the query by using the following steps:
    a. First, you should determine what modality (EHR, notes, images, etc.) are available in the dataset and what modality agents are enabled.
    b. Based on the available modalities, and available modality agents, you should decide what modality agent(s) to use.
    c. After the usage of the modality agents, you must use the SummaryAgent to aggregate the the information from the modality agents
    d. The SummaryAgent will perform the final prediction.


[Important Notes]
Please note that if you want to load any data or model, please load them from metadata's local context variables such as dataset or model. 
Do not generate any path or file name yourself.
If the task is not related to the enabled agents, or the task is beyond your capabilities, you should return "I'm sorry, I can't accomplish that query."
        """
        orchestration_task = system_prompt.format(task=task)
        # Use BaseOrchestrator's execute method with Dataset input
        result_dataset = super().run(orchestration_task, patient_id=patient_id, year=year, institution=institution, diagnosis=diagnosis, time_to_event=time_to_event, volume=volume)

        # Add final orchestration metadata
        final_info = result_dataset.get_metadata_info()

        return result_dataset