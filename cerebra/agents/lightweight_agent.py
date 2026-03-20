# cerebra/agents/lightweight_agent.py
from typing import Dict, Any, List, Optional
from cerebra.engine.factory import create_llm_engine
from cerebra.agents.modules.initializer import Initializer
from cerebra.agents.modules.executor import Executor
import json
import pickle
from llm_output_parser import parse_xml
from cerebra.utils.utils import dict_to_xml_str

from cerebra.utils.metadata import Metadata
import copy
class LightweightAgent:
    """
    Lightweight agent with LLM reasoning + tool calling capabilities.
    No complex planning loop - just: Reason → Select Tool → Execute → Return
    """

    def __init__(
        self,
        agent_name: str,
        llm_engine_name: str = "gpt-4o",
        multimodal: bool = False,
        use_nyu_hipaa: bool = False,
        enabled_tools: List[str] = ["all"],
        verbose: bool = True,
        root_cache_dir: str = "cerebra_cache"
    ):
        self.agent_name = agent_name
        self.llm_engine_name = llm_engine_name
        self.multimodal = multimodal
        self.verbose = verbose
        self.use_nyu_hipaa = use_nyu_hipaa
        # Initialize tools (reuse existing system)
        self.initializer = Initializer(
            agent_name=agent_name,
            enabled_tools=enabled_tools,
            model_string=llm_engine_name,
            verbose=verbose
        )

        # Lightweight executor (no complex planning)
        self.executor = Executor(agent_name, llm_engine_name, max_time=2000)
        self.executor.set_task_cache_dir(f"{root_cache_dir}/{agent_name}")

        # Simple LLM engine for reasoning
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, use_nyu_hipaa=use_nyu_hipaa)
        self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=True, use_nyu_hipaa=use_nyu_hipaa)
        self.llm_engine_fix_code = create_llm_engine(model_string="o3-mini", use_nyu_hipaa=use_nyu_hipaa)

        # Agent-specific prompts
        self.reasoning_prompt = self._get_reasoning_prompt()



    def _get_reasoning_prompt(self) -> str:
        """Get agent-specific reasoning prompt"""
        return """
Determine the optimal next step to accomplish the given task based on the overall analysis, available tools, and previous steps taken.
Task: {task}
Task Analysis: {task_analysis}

Available Tools: {tools}
Tool Descriptions: {tool_metadata}
{memory}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {remaining_steps}

Instructions:
1. Analyze the context thoroughly, including the task, its analysis, any data, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
   - Key objectives from the task analysis
   - Capabilities of available tools
   - Logical progression of problem-solving
   - Outcomes from previous steps
   - Current step count and remaining steps

3. If the task is not clear or need to be updated, you should return update_task as "True".

4. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

5. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards accomplishing the task.

Rules:
- Select only ONE tool for this step.
- The tool name MUST exactly match one from the available tools list: {tools}.
- Avoid redundancy by considering previous steps and building on prior results.
- Always use the fields in the dictionary of the dataset and model, NOT the raw values.
- If a previous step failed, do NOT select the same tool with the same parameters. Either pick a prerequisite tool or adjust the approach.
        """

    def _get_evaluation_prompt(self) -> str:
        evaluation_prompt = """
Task Analysis: {task_analysis}
Tool Used: {selected_tool}
Available Tools: {available_tools}
Tool Parameters: {tool_parameters}
Tool Evaluation Criteria: {tool_evaluation_criteria}
Execution Result for the current step: {execution_result}
Evaluation criteria for the current step: {tool_evaluation_criteria}

You should check whether the tool execution result for the current step meets the evaluation criteria.
If the tool execution result for the current step meets the evaluation criteria, you should return is_tool_completed as true.
If the tool execution result for the current step does not meet the evaluation criteria, you should return is_tool_completed as false.

The task might requires several steps to complete. You should check whether the task is completed based on the execution result of each step.
If the task is completed, you should return is_task_completed as true to save time and resources.
If the task is not completed, you should return is_task_completed as false and provide the next step to complete the task.
        """
        return evaluation_prompt

    def _get_planning_prompt(self) -> str:
        """Get prompt for initial overall planning"""
        planning_prompt = """
        You are a {agent_name} planning agent.

        Goal: {task}
        Available Tools: {tools}
        Tool Descriptions: {tool_metadata}
        Data Context: {data_info}
        Current Execution State (what has already been produced by previous steps): {current_execution_state}
        Task update reasoning: {task_update_reasoning}

        Your task is to create a comprehensive plan to achieve the given goal. Analyze the goal, understand what needs to be accomplished, and create a step-by-step plan using the available tools.

        Consider:
        1. If the goal update reasoning is provided, you should consider the reasoning and propose a new goal accordingly.
        2. What is the ultimate objective?
        3. What intermediate steps are needed?
        4. Which tools are most appropriate for each step?
        5. What dependencies exist between steps?
        6. What data transformations or processing might be needed?
        7. What are the expected outcomes at each step?
        8. You should consider the data context and the available tools to create the plan.
        
        IMPORTANT: Inspect the Current Execution State carefully.
        - If the 'model' dictionary is empty or missing a required key (e.g. 'trained_model'), you MUST include the appropriate training tool as a step BEFORE any inference step.
        - Never plan an inference step if no trained model exists in the execution state.

        Create a detailed plan that will guide the execution process. Each step should be clear, actionable, and specify which tool to use.
        For efficiency, each step should better use different tools. If you use the same tool for multiple steps, you should consider merging the steps into one step.


        Respond with XML format:
        <plan>
            <goal_analysis>Detailed analysis of what needs to be accomplished</goal_analysis>
            <overall_strategy>High-level approach to achieve the goal</overall_strategy>
            <steps>
                <step number="1">
                    <description>What to accomplish in this step</description>
                    <recommended_tool>tool_name</recommended_tool>
                    <expected_outcome>What should result from this step</expected_outcome>
                    <dependencies>Any dependencies on previous steps</dependencies>
                </step>
                <step number="2">
                    <description>What to accomplish in this step</description>
                    <recommended_tool>tool_name</recommended_tool>
                    <expected_outcome>What should result from this step</expected_outcome>
                    <dependencies>Any dependencies on previous steps</dependencies>
                </step>
                <!-- Add more steps as needed -->
            </steps>
            <success_criteria>How to determine if the overall goal has been achieved</success_criteria>
            <potential_challenges>Anticipated challenges and how to address them</potential_challenges>
        </plan>
        """
        return planning_prompt

    def _create_overall_plan(self, task: str, data: Metadata, task_update_reasoning: str = None, execution_context: Metadata = None) -> Dict[str, Any]:
        """Create an overall plan for achieving the goal before starting execution"""

        if self.verbose:
            print(f"\n📋 [{self.agent_name}] Creating overall plan for task: {task}")

        metadata_info = data.get_metadata_info()
        current_execution_state = execution_context.get_metadata_info() if execution_context else {}

        if task_update_reasoning is None:
            task_update_reasoning = ""

        prompt = self._get_planning_prompt().format(
            agent_name=self.agent_name,
            task=task,
            tools=list(self.initializer.available_tools),
            tool_metadata=json.dumps(self.initializer.toolbox_metadata, indent=2),
            data_info=metadata_info,
            current_execution_state=current_execution_state,
            task_update_reasoning=task_update_reasoning,
        )

        max_retries = 5
        attempt = 0
        plan = None
        while attempt < max_retries:
            try:
                response = self.llm_engine(prompt)
                plan = parse_xml(response)
                # Check for minimal success: must have steps and goal_analysis
                if plan and plan.get('steps') and plan.get('goal_analysis'):
                    if self.verbose:
                        print(f"🎯 Goal Analysis: {plan.get('goal_analysis', 'Not provided')}")
                        print(f"📈 Strategy: {plan.get('overall_strategy', 'Not provided')}")
                        # Print planned steps
                        steps = plan.get('steps', {})
                        if isinstance(steps, dict):
                            step_items = []
                            for key, value in steps.items():
                                if key.startswith('step'):
                                    step_items.append((key, value))
                            step_items.sort(key=lambda x: x[0])  # Sort by step key

                            for _, step_info in step_items:
                                for step_info_item in step_info:
                                    step_num = step_info_item.get('@number', 'Unknown')
                                    description = step_info_item.get('description', 'No description')
                                    tool = step_info_item.get('recommended_tool', 'No tool specified')
                                    print(f"  Step {step_num}: {description} (Tool: {tool})")

                        print(f"✅ Success Criteria: {plan.get('success_criteria', 'Not defined')}")
                    return plan
                else:
                    if self.verbose:
                        print(f"⚠️ Planning attempt {attempt+1} failed: Incomplete plan, retrying...")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error in planning attempt {attempt+1}: {str(e)}")
            attempt += 1

        # If all attempts fail, return fallback plan
        if self.verbose:
            print(f"⚠️ Error in planning: All {max_retries} attempts failed.")
        return {
            "goal_analysis": "Error in planning: Unable to generate plan after multiple attempts",
            "overall_strategy": "Fallback to iterative approach",
            "steps": {"step_1": {"description": "Proceed with basic reasoning", "recommended_tool": "any", "expected_outcome": "Basic execution"}},
            "success_criteria": "Task completion without errors",
            "potential_challenges": "Planning system unavailable"
        }


    def reason_and_execute(self, task: str, input_metadata: Metadata, max_iterations: int = 5) -> Metadata:
        """
        Enhanced execution method: Plan first, then iteratively reason, execute, and improve until success.
        """
        if self.verbose:
            print(f"\n🧠 [{self.agent_name}] Starting plan-then-execute approach for task: {task}")
        
        # Step 0: Create overall plan before starting execution
        overall_plan = self._create_overall_plan(task, input_metadata)

        iteration = 0
        previous_attempts = []

        merged_excution_context = copy.deepcopy(input_metadata)

        while iteration < max_iterations:
            iteration += 1

            if self.verbose:
                print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            # Step 1: Reason about the task (with context from overall plan and previous attempts)
            reasoning_result = self._reason_about_task_with_context(task, previous_attempts, merged_excution_context, overall_plan, iteration, max_iterations)
            if self.verbose:
                print(f"💭 Reasoning: {reasoning_result.get('analysis', 'No analysis')}")
                print(f"🔧 Selected Tool: {reasoning_result.get('selected_tool', 'None')}")

            # If no tool is selected, try to regenerate the reasoning_result up to max_iterations
            retry_reasoning_attempts = 0
            max_reasoning_retries = 10 
            while not reasoning_result.get('selected_tool') and retry_reasoning_attempts < max_reasoning_retries:
                if self.verbose:
                    print(f"⚠️ No tool selected, regenerating reasoning (attempt {retry_reasoning_attempts+1}/{max_reasoning_retries})...")
                retry_reasoning_attempts += 1
                reasoning_result = self._reason_about_task_with_context(
                    task, previous_attempts, merged_excution_context, overall_plan, iteration, max_iterations
                )
                if self.verbose:
                    print(f"💭 Reasoning (retry): {reasoning_result.get('analysis', 'No analysis')}")
                    print(f"🔧 Selected Tool (retry): {reasoning_result.get('selected_tool', 'None')}")
            
            # Safety net: if the last 3 iterations all failed, force a plan reset
            # regardless of what the evaluator says (handles cases where evaluator itself is wrong)
            CONSECUTIVE_FAILURE_THRESHOLD = 3
            if len(previous_attempts) >= CONSECUTIVE_FAILURE_THRESHOLD:
                recent = previous_attempts[-CONSECUTIVE_FAILURE_THRESHOLD:]
                if all(not a['evaluation'].get('technical_success', False) for a in recent):
                    failed_tool = recent[-1]['reasoning'].get('selected_tool', 'unknown')
                    last_error = recent[-1]['evaluation'].get('failure_reason', 'Unknown error')
                    last_next_step = recent[-1]['evaluation'].get('next_step', '')
                    task_update_reasoning = (
                        f"The agent has failed {CONSECUTIVE_FAILURE_THRESHOLD} consecutive iterations.\n"
                        f"Last failing tool: {failed_tool}\n"
                        f"Last error: {last_error}\n"
                        f"Evaluator suggested: {last_next_step}\n"
                        f"The current plan is not working. Create a revised plan that addresses the root cause."
                    )
                    if self.verbose:
                        print(f"🔁 Safety net triggered: {CONSECUTIVE_FAILURE_THRESHOLD} consecutive failures. Forcing plan reset.")
                    overall_plan = self._create_overall_plan(task, input_metadata, task_update_reasoning, execution_context=merged_excution_context)
                    reasoning_result = self._reason_about_task_with_context(
                        task, previous_attempts, merged_excution_context, overall_plan, iteration, max_iterations
                    )
                    if self.verbose:
                        print(f"💭 Reasoning (safety net replan): {reasoning_result.get('analysis', 'No analysis')}")
                        print(f"🔧 Selected Tool (safety net replan): {reasoning_result.get('selected_tool', 'None')}")
            
            if not reasoning_result.get('selected_tool'):
                # If still no tool selected after retries, return error
                return Metadata.create_agent_output(
                    status="error",
                    dataset={
                        "message": "No appropriate tool selected",
                        "agent": self.agent_name,
                        "overall_plan": overall_plan,
                        "reasoning": reasoning_result,
                        "iteration": iteration,
                        "previous_attempts": previous_attempts
                    },
                    model={},
                    cache_directory=f"cerebra_cache/{self.agent_name}",
                    agent_name=self.agent_name,
                    status_description=f"No tool selected for task: {task}"
                )
            
            # Step 2: Execute the selected tool
            execution_result = self._execute_tool(
                reasoning_result['selected_tool'],
                reasoning_result.get('tool_parameters', {}),
                merged_excution_context
            )

            if execution_result:
                if isinstance(execution_result, list):
                    for i in range(len(execution_result)):
                        if isinstance(execution_result[i], Metadata):
                            merged_excution_context.merge_metadata(execution_result[i])
                        else:
                            merged_excution_context.merge_metadata(Metadata.create_agent_output(
                                status="failed",
                                dataset={
                                    "error": {
                                        "saved_path": None,
                                        "description": execution_result[i],
                                        "configuration": {}
                                    }
                                },
                                model={},
                                cache_directory=f"cerebra_cache/{self.agent_name}"
                            ))
                else:
                    if isinstance(execution_result, Metadata):
                        merged_excution_context.merge_metadata(execution_result)
                    else:
                        merged_excution_context.merge_metadata(Metadata.create_agent_output(
                            status="failed",
                            dataset={
                                "error": {
                                    "saved_path": None,
                                    "description": execution_result,
                                    "configuration": {}
                                }
                            },
                            model={},
                            cache_directory=f"cerebra_cache/{self.agent_name}"
                        ))

            # import pdb; pdb.set_trace()

            # Step 3: Evaluate the result
            evaluation_result = self._evaluate_execution_result(merged_excution_context, reasoning_result, overall_plan)

            if self.verbose:
                print(f"📊 Next step: {evaluation_result.get('next_step', 'No next step')}")
                print(f"✅ Tool completion: {evaluation_result.get('is_tool_completed', False)}")
                print(f"✅ Task completion: {evaluation_result.get('is_task_completed', False)}")
                print(f"✅ Task not completed reason: {evaluation_result.get('evaluation_reasoning', 'No reason')}")
                print(f"🔁 Needs plan revision: {evaluation_result.get('needs_plan_revision', False)}")

            # Evaluation-driven plan revision: if the evaluator identifies the plan is structurally wrong,
            # reset the overall plan immediately before the next iteration
            if evaluation_result.get('needs_plan_revision', False) and not evaluation_result.get('is_task_completed', False):
                task_update_reasoning = evaluation_result.get('plan_revision_reasoning', 'Plan revision requested by evaluator.')
                if self.verbose:
                    print(f"📋 Evaluator triggered plan revision: {task_update_reasoning}")
                overall_plan = self._create_overall_plan(task, input_metadata, task_update_reasoning, execution_context=merged_excution_context)

            # Record this attempt
            attempt = {
                "iteration": iteration,
                "reasoning": reasoning_result,
                "execution": execution_result,
                "evaluation": evaluation_result
            }
            previous_attempts.append(attempt)

            # Step 4: Check if we should continue or stop
            if evaluation_result.get('is_task_completed', False):
                if self.verbose:
                    print(f"🎉 Task completed successfully in {iteration} iteration(s)!")

                return Metadata.create_agent_output(
                    status="success",
                    dataset={
                        "agent": self.agent_name,
                        "task": task,
                        "overall_plan": overall_plan,
                        "reasoning": reasoning_result,
                        "evaluation": evaluation_result,
                        "iterations_used": iteration,
                        "agent_output": merged_excution_context.get_metadata_info() if merged_excution_context else {}
                    },
                    model={},
                    cache_directory=f"cerebra_cache/{self.agent_name}",
                    agent_name=self.agent_name,
                    status_description=f"Task completed successfully: {task}"
                )

            if self.verbose:
                print(f"⚠️ Iteration {iteration} not successful. Next step: {evaluation_result.get('next_step', 'Unknown')}")
                if iteration < max_iterations:
                    print(f"🔄 Preparing for next iteration...")

        # If we've exhausted all iterations without success
        return Metadata.create_agent_output(
            status="partial_success",
            dataset={
                "message": f"Task not fully completed after {max_iterations} iterations",
                "agent": self.agent_name,
                "overall_plan": overall_plan,
                "best_attempt": previous_attempts[-1] if previous_attempts else None,
                "iterations_used": max_iterations,
                "previous_attempts": previous_attempts,
                "task": task
            },
            model={},
            cache_directory=f"cerebra_cache/{self.agent_name}",
            agent_name=self.agent_name,
            status_description=f"Task partially completed after {max_iterations} iterations: {task}"
        )

    def _reason_about_task_with_context(self, task: str, previous_attempts: List[Dict[str, Any]], excution_context: Metadata, overall_plan: Dict[str, Any], iteration: int, max_iterations: int) -> Dict[str, Any]:
        """Enhanced reasoning that considers previous attempts, their outcomes, and performance metrics"""

        # Build context from overall plan
        plan_context = ""
        if overall_plan:
            plan_context += f"Goal Analysis: {overall_plan.get('goal_analysis', 'Not provided')}\n"
            plan_context += f"Strategy: {overall_plan.get('overall_strategy', 'Not provided')}\n"


            steps = overall_plan.get('steps', {})
            if isinstance(steps, dict):
                step_items = []
                for key, value in steps.items():
                    if key.startswith('step'):
                        step_items.append((key, value))
                step_items.sort(key=lambda x: x[0])  # Sort by step key

                for _, step_info in step_items:
                    for step_info_item in step_info:
                        step_num = step_info_item.get('@number', 'Unknown')
                        description = step_info_item.get('description', 'No description')
                        tool = step_info_item.get('recommended_tool', 'No tool specified')
                        plan_context += f"  Step {step_num}: {description} (Tool: {tool}) \n"

        # Build context from previous attempts
        context_info = ""
        if previous_attempts:
            context_info = "\n\nPrevious Attempts and Results:\n"
            for i, attempt in enumerate(previous_attempts, 1):
                eval_result = attempt.get('evaluation', {})
                context_info += f"Attempt {i}:\n"
                if eval_result:
                    context_info += f"  Tool: {attempt['reasoning'].get('selected_tool', 'Unknown')}\n"
                    context_info += f"  Parameters: {attempt['reasoning'].get('tool_parameters', {})}\n"
                    context_info += f"  Technical Success: {eval_result.get('technical_success', 'Unknown')}\n"
                    context_info += f"  Performance Metrics: {eval_result.get('performance_metrics', 'None')}\n"
                    context_info += f"  Overall Task Completion: {eval_result.get('is_task_completed', False)}\n"
                    context_info += f"  Issues (if any): {eval_result.get('failure_reason', 'None')}\n"
                    context_info += f"  Next Step (if any): {eval_result.get('next_step', 'None')}\n"

            # Add execution output information
            if excution_context:
                execution_metadata = excution_context
                if hasattr(execution_metadata, 'get_metadata_info'):
                    execution_info = execution_metadata.get_metadata_info()
                    context_info += f"  Execution Context:\n"
                    context_info += f"    Status: {execution_info.get('status', 'Unknown')}\n"

                    # Add available datasets from output
                    if execution_info:
                        datasets = execution_info.get('dataset', {})
                        models = execution_info.get('model', {})

                        if datasets:
                            context_info += f"The dataset in the excution context is in the format of the data dictionary.\n"
                            context_info += 'dataset = ' + str(datasets) + '\n'
                    
                        if models:
                            context_info += f"The model in the execution context is in the format of the model dictionary.\n"
                            context_info += 'model = ' + str(models) + '\n'
                    
                    
        # ======================
        # Final prompt construction
        # ======================

        prompt = self.reasoning_prompt.format(
            task_analysis=plan_context,
            memory=context_info,
            step_count=iteration,
            max_step_count=max_iterations,
            remaining_steps=max_iterations - iteration,
            agent_name=self.agent_name,
            tools=list(self.initializer.available_tools),
            tool_metadata=json.dumps(self.initializer.toolbox_metadata, indent=2),
            task=task
        )
        # Add structured output format
        prompt += """
        Respond with XML format.

        Important rules for tool parameters:
        - <param1>, <param2>, <param3>, ... are ONLY placeholder XML tags used to list parameters.
        - Inside each tag, you must provide the actual parameter name and value in the format:

            parameter_name=value

        - The parameter_name MUST exactly match the real argument name required by the selected tool.
        - DO NOT output param1=..., param2=..., etc.
        - Instead output things like:
            train_data_path=...
            train_labels_path=...
        Example:
        <tool_parameters>
            <train_data_path>dataset['train_data']['saved_path']</train_data_path>
            <train_labels_path>dataset['train_labels']['saved_path']</train_labels_path>
        </tool_parameters>

        Format: 

        <response>
            <analysis>Your analysis of the task and data, considering the overall goal, current step, and previous attempts</analysis>
            <current_step_analysis>Which step in the overall goal you are working on and why</current_step_analysis>
            <selected_tool>tool_name</selected_tool>
            <tool_parameters>
                <param1>value1</param1>
                <param2>value2</param2>
            </tool_parameters>
            <plan_alignment>How this action aligns with the overall goal and moves toward the goal</plan_alignment>
            <improvement_reasoning>Explain why this approach should achieve better performance than previous attempts</improvement_reasoning>
            <performance_expectations>What performance metrics do you expect to achieve and why?</performance_expectations>
        </response>
        """
        try:
            response = self.llm_engine(prompt)
            # Parse XML response
            result = parse_xml(response)
            return result
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error in reasoning: {str(e)}")
            return {
                "analysis": f"Error in reasoning: {str(e)}",
                "current_step_analysis": "Unable to determine due to error",
                "selected_tool": self.initializer.available_tools[0] if self.initializer.available_tools else None,
                "tool_parameters": {},
                "plan_alignment": "Fallback due to reasoning error",
                "improvement_reasoning": "Fallback due to reasoning error",
                "performance_expectations": "Unable to set expectations due to reasoning error"
            }

    def _evaluate_execution_result(self, execution_result: Metadata, reasoning_result: Dict[str, Any], overall_plan: Dict[str, Any]) -> Dict[str, Any]:

        """Evaluate if the execution result is successful and meets performance criteria"""

        # Extract dataset directly
        if isinstance(execution_result, list):
            execution_result = execution_result[0]
        execution_result = execution_result.get_metadata_info()

        try:
            tool_evaluation_criteria = self.initializer.toolbox_metadata[reasoning_result.get('selected_tool', 'Unknown')].get('evaluation_criteria', {})
            tool_evaluation_criteria = '\n'.join(f"{k.replace('_', ' ')}: {v}" for k, v in tool_evaluation_criteria.items())
        except Exception as e:
            tool_evaluation_criteria = 'None'
            if self.verbose:
                print(f"⚠️ No evaluation criteria found for tool: {reasoning_result.get('selected_tool', 'Unknown')}")

        # Build plan context for evaluation
        plan_evaluation_context = ""
        if overall_plan:
            plan_evaluation_context = "\n\nOVERALL PLAN EVALUATION CONTEXT:\n"
            plan_evaluation_context += f"Goal Analysis: {overall_plan.get('goal_analysis', 'Not provided')}\n"
            plan_evaluation_context += f"Overall Strategy: {overall_plan.get('overall_strategy', 'Not provided')}\n"
            plan_evaluation_context += f"Success Criteria: {overall_plan.get('success_criteria', 'Not defined')}\n\n"

            # Add step context
            current_step_analysis = reasoning_result.get('current_step_analysis', 'Not provided')
            plan_alignment = reasoning_result.get('plan_alignment', 'Not provided')

            plan_evaluation_context += f"Current Step Being Evaluated: {current_step_analysis}\n"
            plan_evaluation_context += f"Goal Alignment: {plan_alignment}\n\n"

            # Add planned steps for reference
            steps = overall_plan.get('steps', {})
            if isinstance(steps, dict):
                plan_evaluation_context += "PLANNED STEPS (for reference):\n"
                step_items = []
                for key, value in steps.items():
                    if key.startswith('step'):
                        step_items.append((key, value))
                step_items.sort(key=lambda x: x[0])

                for _, step_info in step_items:
                    for step_info_item in step_info:
                        step_num = step_info_item.get('@number', 'Unknown')
                        description = step_info_item.get('description', 'No description')
                        outcome = step_info_item.get('expected_outcome', 'No outcome specified')
                        plan_evaluation_context += f"  Step {step_num}: {description} (Expected: {outcome})\n"

        evaluation_prompt = self._get_evaluation_prompt().format(
            task_analysis=plan_evaluation_context,
            selected_tool=reasoning_result.get('selected_tool', 'Unknown'),
            tool_parameters=reasoning_result.get('tool_parameters', {}),
            available_tools=self.initializer.available_tools,
            execution_result=dict_to_xml_str("excution_results", execution_result),
            tool_evaluation_criteria=tool_evaluation_criteria
        )
        
        # Add plan context to evaluation prompt
        evaluation_prompt += f"Evaluation Guidelines:\n"
        evaluation_prompt += f"1. Does this execution result achieve the expected outcome for the current step?\n"
        evaluation_prompt += f"2. Should we move to the next step in the plan, or retry/refine the current step?\n"
        evaluation_prompt += f"3. Based on the planned steps, how much of the overall task is now complete?\n"
        evaluation_prompt += f"Important: You should consider the task as completed if PLANNED STEPS are finished.\n"
        # Update the response format to include plan progress
        evaluation_prompt += """

Respond with XML format:
<evaluation>
    <is_tool_completed>true/false</is_tool_completed>
    <is_task_completed>true/false</is_task_completed>
    <step_progress>Which step in the plan was just completed (if any) and what's the next step</step_progress>
    <plan_progress_percentage>Estimate percentage of overall plan completed (0-100)</plan_progress_percentage>
    <next_step>If the task is not completed, provide the next step and tool to complete the task</next_step>
    <evaluation_reasoning>Explain your evaluation based on the plan context and execution results</evaluation_reasoning>
    <needs_plan_revision>true/false - set true if the current plan is structurally wrong (e.g. wrong tool order, missing prerequisite step, or repeated failures show the plan cannot work as-is)</needs_plan_revision>
    <plan_revision_reasoning>If needs_plan_revision is true, explain exactly what is wrong with the current plan and what the revised plan should do differently</plan_revision_reasoning>
</evaluation>
        """

        try:
            response = self.llm_engine(evaluation_prompt)
            result = parse_xml(response)
            # Enhanced evaluation based on actual dataset structure
            if isinstance(execution_result, dict):
                # Check technical success
                if execution_result.get('status') != 'success':
                    result['is_task_completed'] = False
                    result['technical_success'] = False
                    result['failure_reason'] = f"Tool execution failed: {execution_result.get('status', 'Unknown error')}"

            # Convert string booleans to actual booleans
            for field in ('is_task_completed', 'is_tool_completed', 'technical_success', 'needs_plan_revision'):
                if isinstance(result.get(field), str):
                    result[field] = result[field].lower() == 'true'

            return result

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error in evaluation: {str(e)}")
            return {
                "is_task_completed": False,
                "is_tool_completed": False,
                "technical_success": False,
                "performance_quality": "poor",
                "step_progress": "Unable to determine due to evaluation error",
                "plan_progress_percentage": "0",
                "evaluation_reasoning": f"Error in evaluation: {str(e)}",
                "failure_reason": f"Error in evaluation: {str(e)}",
                "needs_plan_revision": False,
                "plan_revision_reasoning": ""
            }

    def _execute_tool(self, tool_name: str, parameters: Dict, execution_context: Metadata) -> Metadata:
        """Execute the selected tool with parameters"""
        if tool_name not in self.initializer.available_tools:
            # Create error dataset
            return Metadata.create_agent_output(
                status="error",
                cache_directory=f"cerebra_cache/{self.agent_name}/metadata"
            )


        # Parse and convert parameters to proper types
        parsed_parameters = self._parse_tool_parameters(parameters)

        # Create a proper command with converted parameters
        param_str = ", ".join([
            f"{k}={repr(v)}" if not (isinstance(v, str) and(v.startswith("output[") or v.startswith("dataset[") or v.startswith("model["))) else f"{k}={v}"
            for k, v in parsed_parameters.items()
        ])
        command = f"execution = tool.execute({param_str})"
        if self.verbose:
            print(f"🔧 Executing: {command}")

        # Execute using the existing executor infrastructure
        if isinstance(execution_context, list):
            execution_context = execution_context[-1].get_metadata_info()
        else:
            execution_context = execution_context.get_metadata_info()

        result = self.executor.execute_tool_command(
            self.agent_name,
            tool_name,
            command,
            execution_context=execution_context
        )
        return result

    def _parse_tool_parameters(self, parameters: Dict) -> Dict:
        """Parse and convert tool parameters to proper types"""
        parsed = {}

        for key, value in parameters.items():
            parsed[key] = self._convert_parameter_value(value)

        return parsed

    def _convert_parameter_value(self, value: Any,) -> Any:
        """Convert a parameter value to its proper type"""

        # Handle None values
        if value is None or value == 'None':
            return None

        # If already proper type, return as-is
        if not isinstance(value, str):
            return value

        # Handle dataset references like "dataset['test_data']"
        # Keep them as string references for the executor to handle
        if value.startswith("output[") and value.endswith("]"):
            return value  # Return as-is for executor to handle
        elif value.startswith("dataset[") and value.endswith("]"):
             return value
        elif value.startswith("model[") and value.endswith("]"):
             return value


        # Handle boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'

        # Handle numeric values
        try:
            # Try integer first
            if '.' not in value and not 'e' in value.lower():
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass

        # Handle string literals with quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]  # Remove quotes

        # Return as string if no conversion applies
        return value