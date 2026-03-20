GOAL_ANALYSIS_PROMPT = """
Task: Analyze the given task with accompanying inputs and determine the skills and tools needed to accomplish it effectively.

Available tools: {AVAILABLE_TOOLS}

Metadata for the tools: {TOOLBOX_METADATA}

Data: {DATA}

TASK: {TASK}

Instructions:
1. Carefully read and understand the task and any accompanying inputs.
2. Identify the main objectives.
3. List the specific skills that would be necessary to finish the task comprehensively.
4. Examine the available tools in the toolbox and determine which ones might relevant and useful for finish the task. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to accomplishing the task.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
"""


NEXT_STEP_PROMPT = """
Task: Determine the optimal next sub-task for each agent to accomplish the given task based on the provided analysis, available agents, and previous steps taken.

Context:
Task: {TASK}
Data: {DATA_INFO}
Task Analysis: {TASK_ANALYSIS}

Available Tools:
{AVAILABLE_TOOLS}

Tool Metadata:
{TOOLBOX_METADATA}

Previous Steps and Their Results:
{MEMORY}

Current Step: {STEP_COUNT} in {MAX_STEP_COUNT} steps
Remaining Steps: {REMAINING_STEPS}

Instructions:
1. Analyze the context thoroughly, including the task, its analysis, any data, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
   - Key objectives from the task analysis
   - Capabilities of available tools
   - Logical progression of problem-solving
   - Outcomes from previous steps
   - Current step count and remaining steps

3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards accomplishing the task.

Response Format:
Your response MUST follow this structure:
1. Justification: Explain your choice in detail.
2. Context, Sub-Goal, and Tool: Present the context, sub-goal, and the selected tool ONCE with the following format:

Context: <context>
Sub-Goal: <sub_goal>
Tool Name: <tool_name>

Where:
- <context> MUST include ALL necessary information for the tool to function, structured as follows:
  * Relevant data from previous steps
  * File names or paths created or used in previous steps (list EACH ONE individually)
  * Variable names and their values from previous steps' results
  * Any other context-specific information required by the tool
- <sub_goal> is a specific, achievable objective for the tool, based on its metadata and previous outcomes.
It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
- <tool_name> MUST be the exact name of a tool from the available tools list.

Rules:
- Select only ONE tool for this step.
- The sub-goal MUST directly accomplish the task and be achievable by the selected tool.
- The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
- The tool name MUST exactly match one from the available tools list: {AVAILABLE_TOOLS}.
- Avoid redundancy by considering previous steps and building on prior results.
- Your response MUST conclude with the Context, Sub-Goal, and Tool Name sections IN THIS ORDER, presented ONLY ONCE.
- Include NO content after these three sections.

Example (do not copy, use only as reference):
Justification: [Your detailed explanation here]
Context: data path: "example/image.jpg", Previous detection results: [list of objects]
Sub-Goal: Detect and count the number of specific objects in the image "example/image.jpg"
Tool Name: Object_Detector_Tool

Remember: Your response MUST end with the Context, Sub-Goal, and Tool Name sections, with NO additional content afterwards.
"""


MEMORY_VERIFICATION_PROMPT = """
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given task, considering the potential need for additional tool usage.

Context:
Task: {TASK}
Data: {DATA_INFO}
Available Tools: {AVAILABLE_TOOLS}
Toolbox Metadata: {TOOLBOX_METADATA}
Initial Analysis: {TASK_ANALYSIS}
Memory (tools used and results): {MEMORY}

Detailed Instructions:
1. Carefully analyze the task, initial analysis, and data (if provided):
   - Identify the main objectives of the task.
   - Note any specific requirements or constraints mentioned.
   - If any data is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the task.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to accomplishing the task.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the task?
      - Identify any parts of the task that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

Response Format:

If the memory is complete, accurate, AND verified:
Explanation:
<Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the task has been satisfied.>

Conclusion: STOP

If the memory is incomplete, insufficient, or requires further verification:
Explanation:
<Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the task. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

Conclusion: CONTINUE

IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion.
"""

FINAL_OUTPUT_PROMPT = """
Task: Generate the final output based on the task, data, and tools used in the process.

Context:
Task: {TASK}
Data: {DATA_INFO}
Actions Taken and results:
{MEMORY}

Instructions:
1. Review the task, data, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.
5. {FINAL_OUTPUT_FORMAT}

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the task and the main findings.

2. Detailed Analysis:
   - Break down the process of accomplishing the task step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to accomplishing the task.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Solution to the Task:
   - Directly address the original task with a clear and concise solution.
   - If the task has multiple parts, ensure each part is solved separately.
   - Follow the final output format if it is provided.
"""

DIRECT_OUTPUT_PROMPT = """
Context:
Task: {TASK}
Data: {DATA_INFO}
Initial Analysis:
{TASK_ANALYSIS}
Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the task, data, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the task.

Answer:
"""


TOOL_COMMAND_PROMPT = """
Task: Generate a precise command to execute the selected tool based on the given information.

Task: {TASK}
Data: {DATA_INFO}
Context: {CONTEXT}
Sub-Goal: {SUB_GOAL}
Selected Tool: {TOOL_NAME}
Tool Metadata: {TOOL_METADATA}

Instructions:
1. Carefully review all provided information: the query, image path, context, sub-goal, selected tool, and tool metadata.
2. Analyze the tool's input_types from the metadata to understand required and optional parameters.
3. Construct a command or series of commands that aligns with the tool's usage pattern and addresses the sub-goal.
4. Ensure all required parameters are included and properly formatted.
5. Use appropriate values for parameters based on the given context, particularly the `Context` field which may contain relevant information from previous steps.
6. If multiple steps are needed to prepare data for the tool, include them in the command construction.

Output Format:
Provide your response in the following structure:

Analysis: <analysis>
Command Explanation: <explanation>
Generated Command:
```python
<command>
```

Where:
- <analysis> is a step-by-step analysis of the context, sub-goal, and selected tool to guide the command construction.
- <explanation> is a detailed explanation of the constructed command(s) and their parameters.
- <command> is the Python code to execute the tool, which can be one of the following types:
    a. A single line command with `execution = tool.execute()`.
    b. A multi-line command with complex data preparation, ending with `execution = tool.execute()`.
    c. Multiple lines of `execution = tool.execute()` calls for processing multiple items.

Rules:
1. The command MUST be valid Python code and include at least one call to `tool.execute()`.
2. Each `tool.execute()` call MUST be assigned to the 'execution' variable in the format `execution = tool.execute(...)`.
3. Each `tool.execute()` call MUST be not include any symbols `\n`.
4. For multiple executions, use separate `execution = tool.execute()` calls for each execution.
5. The final output MUST be assigned to the 'execution' variable, either directly from `tool.execute()` or as a processed form of multiple executions.
6. Use the exact parameter names as specified in the tool's input_types.
6. Enclose string values in quotes, use appropriate data types for other values (e.g., lists, numbers).
7. Do not include any code or text that is not part of the actual command.
8. Ensure the command directly addresses the sub-goal and query.
9. Include ALL required parameters, data, and paths to execute the tool in the command itself.
10. If preparation steps are needed, include them as separate Python statements before the `tool.execute()` calls.

Examples (Not to use directly unless relevant):

Example 1 (Single line command):
Analysis: The tool requires an image path and a list of labels for object detection.
Command Explanation: We pass the image path and a list containing "baseball" as the label to detect.
Generated Command:
```python
execution = tool.execute(image="path/to/image", labels=["baseball"])
```

Example 2 (Multi-line command with data preparation):
Analysis: The tool requires an image path, multiple labels, and a threshold for object detection.
Command Explanation: We prepare the data by defining variables for the image path, labels, and threshold, then pass these to the tool.execute() function.
Generated Command:
```python
image = "path/to/image"
labels = ["baseball", "football", "basketball"]
threshold = 0.5
execution = tool.execute(image=image, labels=labels, threshold=threshold)
```

Example 3 (Multiple executions):
Analysis: We need to process multiple images for baseball detection.
Command Explanation: We call the tool for each image path, using the same label and threshold for all.
Generated Command:
```python
execution = tool.execute(image="path/to/image1", labels=["baseball"], threshold=0.5)
execution = tool.execute(image="path/to/image2", labels=["baseball"], threshold=0.5)
execution = tool.execute(image="path/to/image3", labels=["baseball"], threshold=0.5)
```

Some Wrong Examples:
Generated Command:
```python
execution1 = tool.execute(query="...")
execution2 = tool.execute(query="...")
```
Reason: only `execution = tool.execute` is allowed, not `execution1` or `execution2`.

Generated Command:
```python
execution = tool.execute(query="...",
                         data_path="...",
                           ...
)
```
Reason: extra line breaks are not allowed.


Generated Command:
```python
urls = [
    "https://example.com/article1",
    "https://example.com/article2"
]

execution = tool.execute(url=urls[0])
execution = tool.execute(url=urls[1])
```
Reason: The command should process multiple items in a single execution, not separate executions for each item.

Remember: Your response MUST end with the Generated Command, which should be valid Python code including any necessary data preparation steps and one or more `execution = tool.execute(` calls, without any additional explanatory text. The format `execution = tool.execute` must be strictly followed, and the last line must begin with `execution = tool.execute` to capture the final output."""