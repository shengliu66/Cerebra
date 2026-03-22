# agents/base_agent.py

import time
from typing import Any, Dict, Optional, Union
import os
import re
import uuid
import pandas as pd
from cerebra.agents.modules.planner import Planner
from cerebra.agents.modules.memory import Memory
from cerebra.agents.modules.executor import Executor
from cerebra.agents.modules.utils import make_json_serializable_truncated, safe_parse_dict
from cerebra.agents.modules.initializer import Initializer
from cerebra.utils.metadata import Metadata
from cerebra.agents.modules.formatters import PatientIDExtraction
from cerebra.engine.factory import create_llm_engine

class BaseOrchestrator:
    def __init__(
        self,
        llm_engine_name: str,
        orchestrator_name: str,
        enabled_agents : list[str] = ["all"],
        max_steps: int = 10,
        verbose:   bool = True,
        vllm_config_path: str = None,
        root_cache_dir: str = "cerebra_cache"
    ):

        self.max_steps = max_steps
        self.verbose = verbose
        self.llm_engine_name = llm_engine_name

        # Load available agents
        self.available_agents = self._load_agents(enabled_agents)
        self.agent_metadata = self._get_agent_metadata()
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False)

        self.planner = Planner(
            orchestrator_name,
            llm_engine_name,
            available_agents=list(self.available_agents.keys()),  # Pass agent names as available tools
            agent_metadata=self.agent_metadata  # Pass agent metadata as toolbox metadata
        )
        self.memory = Memory(orchestrator_name)
        self.executor = Executor(orchestrator_name, llm_engine_name, max_time=3000)


        # runtime state
        self.task: Optional[str] = None
        self.context: Dict[str,Any] = {}
        self.analysis: Optional[str] = None
        self.input_metadata: Optional[Metadata] = None
        self.current_metadata: Optional[Metadata] = None
        self.result_dict_merged: Dict[str, Any] = {}

        self.root_cache_dir = root_cache_dir
        self.orchestrator_name = orchestrator_name

    def _load_agents(self, enabled_agents: list[str]) -> Dict[str, Any]:
        """Load and instantiate the requested agents."""
        available_agents = {}
        # Import agents dynamically
        agent_configs = {
            "ehr_agent": ("cerebra.agents.ehr_agent", "EHRAgent", True),
            "note_agent": ("cerebra.agents.note_agent", "NoteAgent", True), 
            "analysis_agent": ("cerebra.agents.analysis_agent", "AnalysisAgent", True),
            "data_agent": ("cerebra.agents.data_agent", "DataAgent", False),
            "summary_agent": ("cerebra.agents.summary_agent", "SummaryAgent", True),
            "image_agent": ("cerebra.agents.image_agent", "ImageAgent", True)
        }
        
        for agent_name, (module_path, class_name, needs_llm) in agent_configs.items():
            if "all" in enabled_agents or agent_name in enabled_agents:
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    agent_class = getattr(module, class_name)
                    
                    if needs_llm:
                        available_agents[agent_name] = agent_class(self.llm_engine_name)
                    else:
                        available_agents[agent_name] = agent_class()
                        
                except ImportError:
                    if self.verbose:
                        print(f"⚠️ {class_name} not available")
        if self.verbose:
            print(f"🤖 Loaded agents: {list(available_agents.keys())}")

        return available_agents

    def _get_agent_metadata(self) -> Dict[str, Any]:
        """Get metadata for all loaded agents."""
        metadata = {}

        for agent_name, agent_instance in self.available_agents.items():
            if hasattr(agent_instance, 'register_agent_capabilities'):
                # If agent has a method to register capabilities
                agent_capabilities = agent_instance.register_agent_capabilities()
                metadata.update(agent_capabilities)
            else:
                # Default metadata for agents
                metadata[agent_name] = {
                    "agent_description": f"{agent_name.replace('_', ' ').title()}",
                    "agent_capabilities": ["data_processing", "analysis"],
                    "agent_input_types": {"task": "str", "data": "Dataset"},
                    "agent_output_type": "Dataset with processed results"
                }

        return metadata

    # ——————————————————————————————————————————
    # 1) OBSERVE
    # ——————————————————————————————————————————
    def observe(self, task: str) -> None:
        """
        Initialize the orchestrator with a new task.
        Assign task to the data agent and load the data
        """
        self.task = task

        # Use LLM to detect and extract patient ID from task
        patient_id_prompt = f"""
        Analyze the following task and extract the patient ID if mentioned.
        Task: {self.task}
        """
        if self.patient_id is None:
            try:
                patient_id_response = self.llm_engine(patient_id_prompt, response_format=PatientIDExtraction)
                if patient_id_response.patient_id:
                    self.patient_id = patient_id_response.patient_id
                    if self.verbose:
                        print(f"📋 Extracted patient ID: {self.patient_id}")
                else:
                    self.patient_id = None
                    if self.verbose:
                        print("📋 No patient ID found in task")
            except Exception as e:
                self.patient_id = None
                if self.verbose:
                    print(f"⚠️ Error extracting patient ID: {str(e)}")

        # TODO: load data from data agent, adding a llm to assign sub-goal from task for data agent
        self.data_info = {
            # "dataset_stored_path": '/path/to/dataset',
            "dataset_type": 'raw_dataset',
            "EHR_dataset_description": 'EHR raw data, in sparse matrix format)', #'',
            "EHR_feature_description": 'Diagnosis, Medication, Lab, etc.',#'.',
            "Note_dataset_description": 'Note data, in list of strings for each patient and EHR raw data, in sparse matrix format)',
            "Note_feature_description": 'radiology report, progress note, etc.',
            "Image_dataset_description": 'Image data, in list of strings for each patient',
            "Image_feature_description": 'CT scan, MRI scan, etc.',
            "dataset_len": 1000,
        }#self.available_agents['data_agent'].output.get_dataset_info()

        # set cache dir
        _cache_dir = os.path.join(self.root_cache_dir, self.orchestrator_name)
        self.executor.set_task_cache_dir(_cache_dir)
        # self.memory.clear()

    # ——————————————————————————————————————————
    # 2) ANALYZE
    # ——————————————————————————————————————————
    def analyze(self) -> str:
        """
        Perform initial task analysis (agent breakdown, orchestration strategy, etc.).
        """
        self.analysis = self.planner.analyze_agent_goal(
            self.task,
            self.data_info,
        )

        if self.verbose:
            print(f"\n🔍 ANALYZE:\n{self.analysis}")
        return self.analysis

    # ——————————————————————————————————————————
    # 3) PLAN
    # ——————————————————————————————————————————
    def plan(self, step: int) -> Dict[str, Any]:
        """
        Decide the next action: which agent, sub-goal, and context.
        Returns a dict with keys: "context", "sub_goal", "agent_name".
        """
        raw = self.planner.generate_next_step(
            self.task,
            self.data_info,
            self.analysis,
            self.memory,
            step,
            self.max_steps,
        )

        ctx, sub_goal, agent_name = self.planner.extract_context_subgoal_and_agent(raw)

        if self.verbose:
            print(f"\n➡️ PLAN (step {step}):\n  Context: {ctx}\n  Sub-goal: {sub_goal}\n  Agent: {agent_name}")
        return {"context": ctx, "sub_goal": sub_goal, "agent_name": agent_name}

    # ——————————————————————————————————————————
    # 4) EXECUTE
    # ——————————————————————————————————————————
    def execute(self, plan: Dict[str, Any]) -> Metadata:
        """Execute agent and return Dataset"""

        
        try:
            agent_name = plan["agent_name"]
            if agent_name not in self.available_agents:
                raise ValueError(f"Agent '{agent_name}' not available")
            agent_instance = self.available_agents[agent_name]

            if self.verbose:
                print(f"\n🤖 EXECUTE '{agent_name}':\n  Sub-goal: {plan['sub_goal']}")

            # Get input dataset from context or create from current data
            if hasattr(self, 'current_metadata') and agent_name in ['summary_agent']:
                print("current metadata exists, use it as input metadata")
                input_metadata = self.current_metadata
                from cerebra.agents.data_agent import DataAgent
                data_agent = DataAgent()
            else:
                # Create initial dataset from input data
                from cerebra.agents.data_agent import DataAgent
                data_agent = DataAgent()
                input_metadata = data_agent.run(mode="local", file_paths=self.file_paths.get(agent_name, {}), agent_name=agent_name, patient_id=self.patient_id)
                self.current_metadata = input_metadata
            


            # Execute agent - returns MetaData
            result_dict = agent_instance.run(task=plan["sub_goal"], input_metadata=input_metadata)

            self.result_dict_merged[f"{agent_name}_outputs"] = result_dict
            
            # TODO: this is a hack to get the image path for the summary agent
            image_input_data = data_agent.run(mode="local", file_paths=self.file_paths.get("image_agent", {}), agent_name="image_agent", patient_id=self.patient_id)
            # Load the test_data from the saved_path using pickle
            import pickle
            test_data_path = image_input_data.get_metadata_info()["dataset"]['test_data']['saved_path']
            with open(test_data_path, "rb") as f:
                test_data_loaded = pickle.load(f)
            if isinstance(test_data_loaded, pd.DataFrame):
            # CSV-based image data (e.g. MRI volumes) — no file path available
                self.result_dict_merged["image_path"] = None
            elif isinstance(test_data_loaded, list) and len(test_data_loaded) > 0:
                self.result_dict_merged["image_path"] = test_data_loaded[0]
            else:
                self.result_dict_merged["image_path"] = None
            self.result_dict_merged["patient_id"] = self.patient_id
            self.result_dict_merged["year"] = self.year
            self.result_dict_merged["institution"] = self.institution
            self.result_dict_merged["diagnosis"] = self.diagnosis
            self.result_dict_merged["time_to_event"] = self.time_to_event

            
            # Update current dataset for next agent
            # Merge all output datasets
            self.current_metadata = Metadata.create_agent_output(
                status="success",
                dataset=self.result_dict_merged,
                model={},
                cache_directory=os.path.join(self.root_cache_dir, agent_name),
                agent_name=agent_name
            )
        except Exception as e:
            # Create error Dataset
            error_data = {
                "error": str(e),
                "failed_agent": agent_name,
                "failed_task": plan["sub_goal"]
            }

            self.current_metadata = Metadata.create_agent_output(
                status="error",
                dataset=error_data,
                model={},
                cache_directory=os.path.join(self.root_cache_dir, agent_name),
                agent_name=agent_name
            )
        if self.current_metadata is None:
            if self.verbose:
                print(f"  Result: The agent failed to execute the task")
            return None
        else:
            # Update context & memory
            output_dict = self.current_metadata.get_metadata_info().copy()
            output_dict['dataset'] = 'Available results: ' + ', '.join(list(output_dict['dataset'].keys()))
            self.context.update({agent_name: self.current_metadata.get_metadata_info()})
            self.memory.add_action(len(self.memory.get_actions()) + 1,
                                agent_name, plan["sub_goal"], output_dict)
            return self.current_metadata

    # ——————————————————————————————————————————
    # 5) VERIFY
    # ——————————————————————————————————————————
    def verify(self) -> str:
        """
        Check if the memory is sufficient to STOP or if we should CONTINUE.
        Returns "STOP" or "CONTINUE".
        """
        raw = self.planner.verificate_context(
            self.task,
            self.data_info,
            self.analysis,
            self.memory,
        )
        reason, conclusion = self.planner.extract_conclusion(raw)
        if self.verbose:
            icon = "✅" if conclusion == "STOP" else "🔄"
            print(f"\n🔍 VERIFY: {conclusion} {icon}")
            print(f"Reason: {reason}")
        return conclusion

    def generate_final_output(self, final_output_format: str) -> str:
        """
        Generate the final output for the orchestrator.
        """
        return self.planner.generate_final_output(
            self.task,
            self.inputs.get("meta_info"),
            self.memory,
            final_output_format
        )

    # def _create_output_dataset(self, orchestrator_output: Dict[str, Any], task: str) -> Metadata:
    #     """Create Dataset object from orchestrator output"""

    #     # Define the output structure
    #     output_data = {
    #         "orchestration_result": orchestrator_output,
    #         "source_task": task,
    #         "orchestrator_name": self.orchestrator_name,
    #         "agents_used": list(self.context.keys())
    #     }

    #     feature_descriptions = {
    #         "orchestration_result": f"Final result from {self.orchestrator_name}",
    #         "source_task": "Original task that was orchestrated",
    #         "orchestrator_name": "Name of orchestrator that processed the task",
    #         "agents_used": "List of agents that were executed"
    #     }

        return Metadata.create_agent_output(
            status="success",
            output_dataset={
                "final_orchestration": {
                    "saved_path": "",
                    "description": f"{self.orchestrator_name} orchestration result for: {task}",
                    "configuration": {"agents_used": list(self.context.keys())}
                }
            },
            cache_directory=os.path.join(self.root_cache_dir, self.orchestrator_name)
        )


    # ——————————————————————————————————————————
    # COMPOSED RUN LOOP
    # ——————————————————————————————————————————
    def run(self, task: str, patient_id: str = None, year: int = 1, institution: str = "NYU", diagnosis: bool = False, time_to_event: bool = False, volume: bool = False, file_paths: Dict[str, Dict[str, str]] = None) -> Union[Dict[str, Any], Metadata]:
        """
        Full Observe → Analyze → [Plan → Execute → Verify]* loop for orchestrating agents.
        Returns Dataset if input was Dataset, otherwise returns Dict for backward compatibility.
        """
        self.patient_id = patient_id
        self.year = year
        self.institution = institution
        self.diagnosis = diagnosis
        self.time_to_event = time_to_event
        self.volume = volume
        self.file_paths = file_paths or {}
        # OBSERVE first
        self.observe(task)

        # ANALYZE once
        self.analyze()

        for step in range(1, self.max_steps + 1):

            plan = self.plan(step)
            if not plan["agent_name"]:
                if self.verbose: print("⚠️ No agent selected, stopping.")
                break

            # optional STOP as an agent name
            if plan["agent_name"].upper() == "STOP":
                if self.verbose: print("🛑 Received STOP signal.")
                break

            self.execute(plan)
            if self.verify() == "STOP":
                break

        # output_data = self.context

        # feature_descriptions = {
        #     **{f"{agent_name}": f"Results from {agent_name.replace('_', ' ').title()} agent"
        #        for agent_name in output_data.keys() if isinstance(output_data[agent_name], Metadata)}
        # }

        return Metadata.create_agent_output(
            status="success",
            dataset={
                "final_orchestration": {
                    "saved_path": "",
                    "description": f"{self.orchestrator_name} orchestration result for: {task}",
                    "configuration": {"agents_used": list(self.context.keys())}
                }
            },
            model={},
            agent_name=self.orchestrator_name,
            cache_directory=os.path.join(self.root_cache_dir, self.orchestrator_name)
        )


# Backward compatibility alias
BaseAgent = BaseOrchestrator
