import os
import re
from PIL import Image
from typing import Dict, Any, List, Tuple, Union

from cerebra.engine.factory import create_llm_engine
from cerebra.agents.modules.memory import Memory
from cerebra.agents.modules.formatters import QueryAnalysis, NextStep, MemoryVerification, FinalOutput

class Planner:
    def __init__(self, agent_name: str, llm_engine_name: str, agent_metadata: dict = None, available_agents: List = None, verbose: bool = False):
        self.agent_name = agent_name
        self.llm_engine_name = llm_engine_name
        self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=True)
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False)
        self.agent_metadata = agent_metadata if agent_metadata is not None else {}
        self.available_agents = available_agents if available_agents is not None else []
        self.init_prompt()
        self.verbose = verbose

    def init_prompt(self) -> str:
        from cerebra.agents.prompts.super_agent import GOAL_ANALYSIS_PROMPT, NEXT_STEP_PROMPT, MEMORY_VERIFICATION_PROMPT, FINAL_OUTPUT_PROMPT, DIRECT_OUTPUT_PROMPT
        self.goal_analysis_prompt = GOAL_ANALYSIS_PROMPT
        self.next_step_prompt = NEXT_STEP_PROMPT
        self.memory_verification_prompt = MEMORY_VERIFICATION_PROMPT
        self.final_output_prompt = FINAL_OUTPUT_PROMPT
        self.direct_output_prompt = DIRECT_OUTPUT_PROMPT

    def analyze_agent_goal(self, task: str, data_info: str) -> str:

        task_prompt = self.goal_analysis_prompt.format(
            AGENTS_METADATA=self.agent_metadata,
            AVAILABLE_AGENTS=self.available_agents,
            TASK=task,
            DATA_INFO=data_info
        )

        self.task_analysis = self.llm_engine(task_prompt, response_format=QueryAnalysis)

        return str(self.task_analysis).strip()

    def extract_context_subgoal_and_agent(self, response: Any) -> Tuple[str, str, str]:

        def normalize_agent_name(agent_name: str) -> str:
            # Normalize the agent name to match the available agents
            for agent in self.available_agents:
                if agent.lower() in agent_name.lower():
                    return agent
            return "No matched agent given: " + agent_name

        try:
            if isinstance(response, NextStep):
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                agent_name = response.agent_name.strip()
            else:
                text = response.replace("**", "")

                # Pattern to match the exact format
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Agent Name:\s*(.*?)(?=\n\n|\Z)"

                # Find all matches
                matches = re.findall(pattern, text, re.DOTALL)

                # Return the last match (most recent/relevant)
                context, sub_goal, agent_name = matches[-1]
                context = context.strip()
                sub_goal = sub_goal.strip()
            agent_name = normalize_agent_name(agent_name)
        except Exception as e:
            print(f"Error extracting context, sub-goal, and agent name: {str(e)}")
            return None, None, None

        return context, sub_goal, agent_name

    def generate_next_step(self, task: str, data_info: str, task_analysis: str, memory: Memory, step_count: int, max_step_count: int) -> Any:
        prompt_generate_next_step = self.next_step_prompt.format(
            TASK=task,
            DATA_INFO=data_info,
            TASK_ANALYSIS=task_analysis,
            AVAILABLE_AGENTS=self.available_agents,
            AGENTS_METADATA=self.agent_metadata,
            MEMORY=memory.get_actions(),
            STEP_COUNT=step_count,
            MAX_STEP_COUNT=max_step_count,
            REMAINING_STEPS=max_step_count - step_count
        )
        next_step = self.llm_engine(prompt_generate_next_step, response_format=NextStep)
        return next_step


    def verificate_context(self, task: str, data_info: str, task_analysis: str, memory: Memory) -> Any:
        input_data = self.memory_verification_prompt.format(
            TASK=task,
            DATA_INFO=data_info,
            TASK_ANALYSIS=task_analysis,
            AVAILABLE_AGENTS=self.available_agents,
            AGENTS_METADATA=self.agent_metadata,
            MEMORY=memory.get_actions(),
        )

        stop_verification = self.llm_engine(input_data, response_format=MemoryVerification)

        return stop_verification

    def extract_conclusion(self, response: Any) -> str:
        if isinstance(response, MemoryVerification):
            analysis = response.analysis
            stop_signal = response.stop_signal
            if stop_signal:
                return analysis, 'STOP'
            else:
                return analysis, 'CONTINUE'
        else:
            analysis = response
            pattern = r'conclusion\**:?\s*\**\s*(\w+)'
            matches = list(re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
            # if match:
            #     conclusion = match.group(1).upper()
            #     if conclusion in ['STOP', 'CONTINUE']:
            #         return conclusion
            if matches:
                conclusion = matches[-1].group(1).upper()
                if conclusion in ['STOP', 'CONTINUE']:
                    return analysis, conclusion

            # If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
            if 'stop' in response.lower():
                return analysis, 'STOP'
            elif 'continue' in response.lower():
                return analysis, 'CONTINUE'
            else:
                print("No valid conclusion (STOP or CONTINUE) found in the response. Continuing...")
                return analysis, 'CONTINUE'

    def generate_final_output(self, task: str, data_info: str, memory: Memory, final_output_format: str) -> str:

        prompt_generate_final_output = self.final_output_prompt

        input_data = prompt_generate_final_output.format(
            TASK=task,
            DATA_INFO=data_info,
            MEMORY=memory.get_actions(),
            FINAL_OUTPUT_FORMAT=final_output_format
        )

        final_output = self.llm_engine(input_data, response_format=FinalOutput)

        return final_output


    def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = self.get_image_info(image)

        prompt_generate_final_output = self.direct_output_prompt

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        final_output = self.llm_engine_mm(input_data)

        return final_output
