from pydantic import BaseModel


# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    available_data: str
    relevant_agents: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.concise_summary}

Available Data:
{self.available_data}

Relevant Agents:
{self.relevant_agents}

Additional Considerations:
{self.additional_considerations}
"""

# Planner: NextStep
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    agent_name: str

# Executor: MemoryVerification
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str

class FinalOutput(BaseModel):
    analysis: str
    solution: str


class PatientIDExtraction(BaseModel):
    patient_id: str

# For generating SQL query
class SqlQuery(BaseModel):
    sql_query: str
    explanation: str

# For generating table and column description for Dataset
class TableAndColumnDescription(BaseModel):
    table_description: str
    column_description_dict: str  #TODO: This should be a dict. Need to somehow enforce this

class SaliencyMapAnalysis(BaseModel):
    highlighted_regions: str
    abnormalities: str
    explanation: str

class ModeratorResponse(BaseModel):
    decision: str
    note_expert: str
    image_expert: str
    ehr_expert: str

class ExpertResponse(BaseModel):
    solution: str
    explanation: str

class FinalJudgement(BaseModel):
    risk: float
    explanation: str

# Chat-related models
class ChatResponse(BaseModel):
    """Structured chat response with citations."""
    answer: str  # The actual answer
    evidence_references: list[str]  # Citations like "[NOTE-0]", "[EHR-5]"
    confidence: str  # "high", "medium", "low"
    follow_up_suggestions: list[str]  # Suggested follow-up questions
    needs_evidence_highlight: bool  # True if user asked for original text/context
