import os
import importlib
import re
from typing import Dict, Any, List, Union
from datetime import datetime

from cerebra.engine.factory import create_llm_engine
from cerebra.agents.modules.formatters import ToolCommand

import signal
from typing import Dict, Any, List, Optional
import traceback
import logging
logger = logging.getLogger(__name__)
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class Executor:
    def __init__(self, agent_name: str, llm_engine_name: str,  num_threads: int = 1, max_time: int = 120, max_output_length: int = 100000, verbose: bool = False):
        self.agent_name = agent_name
        self.llm_engine_name = llm_engine_name
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose

        self.init_prompt()

    def init_prompt(self) -> str:
        from cerebra.agents.prompts.super_agent import TOOL_COMMAND_PROMPT

        self.tool_command_prompt = TOOL_COMMAND_PROMPT

    def set_task_cache_dir(self, task_cache_dir):
        if task_cache_dir:
            self.task_cache_dir = task_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.task_cache_dir = os.path.join(self.root_cache_dir, self.agent_name, timestamp)
        os.makedirs(self.task_cache_dir, exist_ok=True)

    def generate_tool_command(self, task: str, data_info: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any]) -> Any:

        prompt_generate_tool_command = self.tool_command_prompt.format(
            TASK=task,
            DATA_INFO=data_info,
            CONTEXT=context,
            SUB_GOAL=sub_goal,
            TOOL_NAME=tool_name,
            TOOL_METADATA=tool_metadata,
        )
        llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False)
        tool_command = llm_generate_tool_command(prompt_generate_tool_command, response_format=ToolCommand)

        return tool_command

    def extract_explanation_and_command(self, response: Any) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove leading and trailing whitespace and triple backticks
            return re.sub(r'^```python\s*', '', code).rstrip('```').strip()

        if isinstance(response, ToolCommand):
            analysis = response.analysis.strip()
            explanation = response.explanation.strip()
            command = response.command.strip()
        else:
            # Extract analysis
            analysis_pattern = r"Analysis:(.*?)Command Explanation"
            analysis_match = re.search(analysis_pattern, response, re.DOTALL)
            analysis = analysis_match.group(1).strip() if analysis_match else "No analysis found."
            # Extract explanation
            explanation_pattern = r"Command Explanation:(.*?)Generated Command"
            explanation_match = re.search(explanation_pattern, response, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."
            # Extract command
            command_pattern = r"Generated Command:.*?```python\n(.*?)```"
            command_match = re.search(command_pattern, response, re.DOTALL)
            command = command_match.group(1).strip() if command_match else "No command found."

        command = normalize_code(command)

        return analysis, explanation, command

    def execute_tool_command(self, agent_name: str, tool_name: str, command: str, execution_context: Dict[str, Any] = None) -> Any:
        """
        Execute a tool command with timeout protection. If execution exceeds max_time seconds,
        the function will be interrupted and return a timeout message.

        Args:
            agent_name (str): Name of the agent
            tool_name (str): Name of the tool to execute
            command (str): Command string containing tool.execute() calls
            execution_context (Dict[str, Any]): Context containing data variables (optional)

        Returns:
            Any: List of execution results or error message
        """
        def split_commands(command: str) -> List[str]:
            # Use regex to find all tool.execute() commands and their surrounding code
            pattern = r'.*?execution\s*=\s*tool\.execute\([^\n]*\)\s*(?:\n|$)'
            blocks = re.findall(pattern, command, re.DOTALL)
            return [block.strip() for block in blocks if block.strip()]

        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            # Set up the timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_time)

            try:
                # Execute the block in the local context
                print("🧪 Executing block with local context variables:", list(local_context.keys()))
                exec(block, globals(), local_context)
                result = local_context.get('execution')
                signal.alarm(0)  # Disable the alarm
                return result
            except TimeoutError:
                return f"Execution timed out after {self.max_time} seconds"
            finally:
                signal.alarm(0)  # Ensure alarm is disabled even if other exceptions occur

        # Import the tool module and instantiate it
        module_name = f"tools.{agent_name}.{tool_name.lower().replace('_tool', '')}.tool"

        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)


            # Get the tool class
            tool_class = getattr(module, tool_name)

            # Check if the tool requires an LLM engine
            # NOTE may need to refine base.py and tool.py to handle this better
            if getattr(tool_class, 'require_llm_engine', False):
                # Instantiate the tool with the model_string
                tool = tool_class(model_string=self.llm_engine_name)
            else:
                # Instantiate the tool without model_string for tools that don't require it
                tool = tool_class()

            # Set the custom output directory
            # NOTE: May have a better way to handle this
            tool.set_custom_output_dir(self.task_cache_dir)

            # Split the command into blocks, execute each one and store execution results
            command_blocks = split_commands(command)
            executions = []

            # if self.agent_name == "ml_agent":
            #         import pdb; pdb.set_trace()

            for block in command_blocks:
                # Create a local context to safely execute the block
                local_context = {'tool': tool}

                # Add execution context data if provided
                if execution_context:
                    local_context.update(execution_context)

                # Execute the block with timeout protection
                result = execute_with_timeout(block, local_context)

                if result is not None:
                    executions.append(result)
                else:
                    executions.append(f"No execution captured from block: {block}")

            # Return all the execution results
            return executions
        except Exception as e:
            import traceback
            logger.error(f"Error during tool execution: {str(e)}")
            logger.error(traceback.format_exc())
            return [str(e)]