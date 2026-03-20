# agents/base_agent.py

import time
from typing import Any, Dict, Optional, Union
import os

from cerebra.agents.modules.planner import Planner
from cerebra.agents.modules.memory import Memory
from cerebra.agents.modules.executor import Executor
from cerebra.agents.modules.utils import make_json_serializable_truncated, safe_parse_dict
from cerebra.agents.modules.initializer import Initializer
from cerebra.utils.metadata import Metadata

class BaseAgent:
    def __init__(
        self,
        llm_engine_name: str,
        agent_name: str,
        enabled_tools : list[str] = ["all"],
        max_steps: int = 10,
        max_time:  int = 300,
        verbose:   bool = True,
        vllm_config_path: str = None,
        root_cache_dir: str = "cerebra_cache"
    ):
        # **all agents share the same Planner/Memory/Executor implementations**
        self.initializer = Initializer(
            agent_name=agent_name,
            enabled_tools=enabled_tools,
            model_string=llm_engine_name,
            verbose=verbose,
            vllm_config_path=vllm_config_path,
        )
        self.planner  = Planner(agent_name, llm_engine_name, available_tools=self.initializer.available_tools, toolbox_metadata=self.initializer.toolbox_metadata)
        self.memory   = Memory(agent_name)
        if agent_name == "ml_agent":
            self.executor = Executor(agent_name, llm_engine_name, max_time=3000)
        else:
            self.executor = Executor(agent_name, llm_engine_name)

        self.max_steps = max_steps
        self.max_time  = max_time
        self.verbose   = verbose

        # runtime state
        self.task    : Optional[str] = None
        self.inputs  : Dict[str,Any] = {}
        self.context : Dict[str,Any] = {}
        self.analysis: Optional[str] = None
        self.input_dataset: Optional[Dataset] = None

        self.agent_name = agent_name
        self.root_cache_dir = root_cache_dir

    # ——————————————————————————————————————————
    # 1) OBSERVE
    # ——————————————————————————————————————————
    def observe(self, task: str, inputs: Union[Dict[str, Any], Dataset]) -> None:
        """
        Initialize the agent with a new task and inputs (Dataset or dict for backward compatibility).
        """
        self.task = task

        # Handle Dataset input
        if isinstance(inputs, Dataset):
            self.input_dataset = inputs
            dataset_info = inputs.get_dataset()
            self.inputs = dataset_info["dataset"]
            if self.verbose:
                print(f"\n📝 OBSERVE: task='{task}', dataset_type={dataset_info['dataset_type']}, dataset_len={dataset_info['dataset_len']}")
        else:
            # Backward compatibility - handle dict input
            self.inputs = inputs.copy() if inputs else {}
            self.input_dataset = None
            if self.verbose:
                print(f"\n📝 OBSERVE: task='{task}', inputs={inputs.get('meta_info') if inputs else 'None'}")

        self.context = {'data': self.inputs.copy()}
        # set cache dir
        _cache_dir = os.path.join(self.root_cache_dir, self.agent_name)
        self.executor.set_task_cache_dir(_cache_dir)
        # self.memory.clear()

    # ——————————————————————————————————————————
    # 2) ANALYZE
    # ——————————————————————————————————————————
    def analyze(self) -> str:
        """
        Perform initial task analysis (skills, tool breakdown, etc.).
        """
        self.analysis = self.planner.analyze_agent_goal(
            self.task,
            self.inputs.get("meta_info"),
        )

        if self.verbose:
            print(f"\n🔍 ANALYZE:\n{self.analysis}")
        return self.analysis

    # ——————————————————————————————————————————
    # 3) PLAN
    # ——————————————————————————————————————————
    def plan(self, step: int) -> Dict[str, Any]:
        """
        Decide the next action: which tool, sub-goal, and context.
        Returns a dict with keys: "context", "sub_goal", "tool_name".
        """
        raw = self.planner.generate_next_step(
            self.task,
            self.inputs.get("meta_info"),
            self.analysis,
            self.memory,
            step,
            self.max_steps,
        )

        ctx, sub_goal, tool = self.planner.extract_context_subgoal_and_tool(raw)

        if self.verbose:
            print(f"\n➡️ PLAN (step {step}):\n  Context: {ctx}\n  Sub-goal: {sub_goal}\n  Tool: {tool}")
        return {"context": ctx, "sub_goal": sub_goal, "tool_name": tool}

    # ——————————————————————————————————————————
    # 4) EXECUTE
    # ——————————————————————————————————————————
    def execute(self, plan: Dict[str, Any]) -> Any:
        """
        Given a plan dict, generate the tool command, run it, update context & memory.
        Returns the raw result.
        """
        tool = plan["tool_name"]
        if tool not in self.planner.available_tools:
            raise ValueError(f"Tool '{tool}' not available")

        # generate command
        cmd_payload = self.executor.generate_tool_command(
            self.task,
            self.inputs.get("meta_info"),
            plan["context"],
            plan["sub_goal"],
            tool,
            self.planner.toolbox_metadata.get(tool, {}),
        )

        analysis, explanation, command = self.executor.extract_explanation_and_command(cmd_payload)
        if self.verbose:
            print(f"\n🛠️ EXECUTE '{tool}':\n  Analysis: {analysis}\n  Explanation: {explanation}\n  Command: {command}")

        # run it with enhanced context
        raw = self.executor.execute_tool_command(
            self.agent_name,
            tool,
            command,
            execution_context=self.context  # Pass the agent's context
        )

        result = make_json_serializable_truncated(raw)
        if self.verbose:
            print(f"  Result: {result}")

        # update context & memory
        self.context.update({tool: result})
        self.memory.add_action(len(self.memory.get_actions()) + 1,
                               tool, plan["sub_goal"], command, result)
        return result

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
            self.inputs.get("meta_info"),
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
        Generate the final output for the agent.
        """
        return self.planner.generate_final_output(
            self.task,
            self.inputs.get("meta_info"),
            self.memory,
            final_output_format
        )

    def _create_output_dataset(self, agent_output: Dict[str, Any], task: str) -> Dataset:
        """Create Dataset object from agent output"""

        # Define the output structure
        output_data = {
            "result": agent_output,
            "source_task": task,
            "processing_agent": self.agent_name
        }

        feature_descriptions = {
            "result": f"Processed output from {self.agent_name}",
            "source_task": "Original task that generated this output",
            "processing_agent": "Name of agent that processed the data"
        }

        return Metadata.create_agent_output(
            processed_data=output_data,
            description=f"{self.agent_name} output for: {task}",
            feature_descriptions=feature_descriptions,
            cache_directory=os.path.join(self.root_cache_dir, self.agent_name)
        )

    # ——————————————————————————————————————————
    # COMPOSED RUN LOOP
    # ——————————————————————————————————————————
    def run(self, task: str, data: Union[Dict[str, Any], Metadata], final_output_format: str = '') -> Union[Dict[str, Any], Metadata]:
        """
        Full Observe → Analyze → [Plan → Execute → Verify]* loop.
        Returns Metadata if input was Metadata, otherwise returns Dict for backward compatibility.
        """
        # Determine return type based on input type
        return_metadata = isinstance(data, Metadata)

        # OBSERVE first
        self.observe(task, data)

        # ANALYZE once
        self.analyze()

        start = time.time()
        for step in range(1, self.max_steps + 1):
            if (time.time() - start) > self.max_time:
                if self.verbose: print("⏰ Reached max_time, stopping.")
                break

            plan = self.plan(step)
            if not plan["tool_name"]:
                if self.verbose: print("⚠️ No tool selected, stopping.")
                break

            # optional STOP as a tool name
            if plan["tool_name"].upper() == "STOP":
                if self.verbose: print("🛑 Received STOP signal.")
                break

            self.execute(plan)
            if self.verify() == "STOP":
                break

        final_output = self.generate_final_output(final_output_format)
        final_output_dict = safe_parse_dict(final_output.solution)

        # Return appropriate format based on input type
        if return_metadata:
            # Create and return Metadata object
            if isinstance(final_output_dict, dict):
                return self._create_output_dataset(final_output_dict, task)
            else:
                # Handle case where final_output_dict is not a dict
                return self._create_output_dataset({"output": final_output_dict}, task)
        else:
            # Return dict for backward compatibility
            if isinstance(final_output_dict, dict):
                return final_output_dict
            else:
                return {}
