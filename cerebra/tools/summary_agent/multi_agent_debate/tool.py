from typing import Dict, Any, List, Optional
from cerebra.tools.base import BaseTool
from cerebra.utils.metadata import Metadata
from datetime import datetime
import json
import os
import numpy as np
import re
from pydantic import BaseModel, Field


class Multi_Agent_Debate_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name="Multi_Agent_Debate_Tool",
            tool_description="Simple debate-based ensemble for multi-modal risk prediction",
            tool_version="2.0.0",
            input_types={
                "metadata_info": "dict - Metadata information from multiple agents containing predictions, risk scores, and evidence",
                "task": "str - The prediction task description",
                "extracted_evidence": "dict - Optional extracted evidence from raw data"
            },
            output_type="dict - Debate results with final risk score and reasoning",
            demo_commands=[
                {
                    "command": "result = tool.execute(metadata_info=input_metadata.get_metadata_info(), task='Predict dementia risk')",
                    "description": "Run simple debate for dementia risk prediction"
                }
            ],
            evaluation_criteria={
                "debate_quality": "Does the debate produce well-reasoned risk scores?",
                "evidence_based": "Is the decision based on actual evidence?",
                "transparency": "Is the reasoning process clear?"
            }
        )
    
    def obtain_dementia_criteria(self) -> str:
        """Obtain dementia criteria from the literature."""
        return """
        AD / ADRD (Alzheimer’s disease and related dementias)
        Includes diagnoses such as: Vascular dementia, Dementia due to other underlying diseases, with or without behavioral disturbance, Unspecified dementia, Amnestic disorders caused by known physiological conditions, Progressive supranuclear palsy, Alzheimer’s disease, Frontotemporal dementias (including Pick’s disease and other variants), Dementia with Lewy bodies, Degenerative diseases of the nervous system, Age-related (senile) degeneration of the brain

        Mild Cognitive Impairment (MCI)
        Includes conditions such as: Mild cognitive impairment of uncertain or unknown cause, Corticobasal degeneration

        Dementia-Related Medications: Common medications used in the treatment of dementia symptoms, including: Donepezil, Galantamine, Memantine, Rivastigmine, Tacrine
        """

    def _extract_agent_data(self, metadata_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract predictions, risk scores, and evidence from metadata."""
        agent_data = {}
        dataset_section = metadata_info.get("dataset", {})
        
        for agent_key, agent_content in dataset_section.items():
            if not isinstance(agent_content, dict) or not agent_key.endswith("_outputs"):
                continue
                
            agent_name = agent_key.replace("_outputs", "")
            agent_content = agent_content.get("dataset", {})
            
            # Extract prediction
            prediction = 0.5
            if "prediction" in agent_content:
                pred_entry = agent_content["prediction"]
                if isinstance(pred_entry, dict) and "configuration" in pred_entry:
                    pred_value = pred_entry["configuration"].get("value", [])
                    prediction = pred_value[0] if pred_value else 0.5
            
            # Extract risk score
            risk_score = 0.5
            if "risk_score" in agent_content:
                risk_entry = agent_content["risk_score"]
                if isinstance(risk_entry, dict) and "configuration" in risk_entry:
                    risk_value = risk_entry["configuration"].get("value", [])
                    risk_score = risk_value[0] if risk_value else 0.5
            
            # Extract evidence
            evidence = []
            if "evidence" in agent_content:
                evidence_entry = agent_content["evidence"]
                if isinstance(evidence_entry, dict) and "configuration" in evidence_entry:
                    evidence_value = evidence_entry["configuration"].get("value", [])
                    if evidence_value and isinstance(evidence_value[0], list):
                        evidence = evidence_value[0]
                    else:
                        evidence = evidence_value if evidence_value else []
            
            agent_data[agent_name] = {
                "prediction": prediction,
                "risk_score": risk_score,
                "evidence": evidence
            }
            
        return agent_data
    
    def _format_evidence_for_debate(self, extracted_evidence: Optional[Dict[str, Any]], 
                                     available_modalities: List[str]) -> Dict[str, str]:
        """Format evidence from agent data and extracted evidence for debate prompts.
        
        Args:
            extracted_evidence: Optional dict with agent and raw evidence
            available_modalities: List of available modalities to process
        """
        evidence_dict = {}
        
        for modality in available_modalities:
            # Get extracted evidence if available
            extracted = extracted_evidence or {}
            agent_evidence = None
            raw_evidence = None
            
            if extracted.get('agent_evidence', {}) is not None:
                agent_evidence = extracted.get('agent_evidence', {}).get(modality, [])
            
            if extracted.get('raw_evidence', {}) is not None:
                raw_evidence = extracted.get('raw_evidence', {}).get(modality)
            
            # Format evidence string
            ev_str = ""
            if agent_evidence: 
                if isinstance(agent_evidence, dict):
                    agent_evidence = agent_evidence.get('volume', [])
                elif isinstance(agent_evidence, list):
                    agent_evidence = agent_evidence
                ev_str += f"\n\nExtracted evidence using attention weights or feature importance from the model:\n"
                ev_str += f"{', '.join(agent_evidence)}"
            
            if raw_evidence and isinstance(raw_evidence, dict):
                ev_str += f"\n\nLLM-Extracted evidence from the raw data:\n"
                ev_str += f"Key Findings: {raw_evidence.get('key_findings', '')}\n"
                ev_str += f"Positive: {', '.join(raw_evidence.get('positive_evidence', []))}\n"
                # ev_str += f"Negative: {', '.join(raw_evidence.get('negative_evidence', []))}"
            
            evidence_dict[modality] = ev_str if ev_str else "No evidence available"
        
        return evidence_dict

    def _run_debate_ensemble(self, modality_scores: Dict[str, float], 
                            agent_data: Dict[str, Any], extracted_evidence: Optional[Dict[str, Any]], 
                            year: int, min_threshold: float = 0.025) -> Dict[str, Any]:
        """Simple debate: highest scorer argues, opposition responds with final score.
        
        Args:
            modality_scores: Dict of available modality scores (e.g., {'image': 0.7, 'note': 0.5})
            agent_data: Agent data dictionary
            extracted_evidence: Optional extracted evidence
            year: Prediction year
            min_threshold: Minimum threshold for debate
        """
        # Filter out None values and invalid scores
        scores = {k: v for k, v in modality_scores.items() if v is not None}
        
        if not scores:
            return {
                'final_score': 0.5,
                'method': 'default (no valid scores)',
                'reasoning': 'No valid modality scores available',
                'debate_transcript': None
            }
        
        # If only one modality, return its score
        if len(scores) == 1:
            single_mod = list(scores.keys())[0]
            return {
                'final_score': scores[single_mod],
                'method': f'single_modality_{single_mod}',
                'reasoning': f'Only {single_mod} modality available with score {scores[single_mod]:.4f}',
                'debate_transcript': None
            }
        
        highest_mod_name = max(scores, key=scores.get)
        highest_score = scores[highest_mod_name]
        
        # If all scores too low, return mean
        if highest_score < min_threshold:
            return {
                'final_score': np.mean(list(scores.values())),
                'method': 'mean (scores too low)',
                'reasoning': f'All scores below {min_threshold}',
                'debate_transcript': None
            }
        
        other_modalities = [(k, v) for k, v in scores.items() if k != highest_mod_name]
        available_modalities = list(scores.keys())
        
        # Format evidence for prompts (only for available modalities)
        evidence_dict = self._format_evidence_for_debate(extracted_evidence, available_modalities)
        

        # Step 1: Highest scorer argues
        prompt_proposal = self._create_debate_prompt_highest(
            highest_mod_name, highest_score, other_modalities, evidence_dict
        )
        
        # Pydantic model for argument
        class ProposalArgument(BaseModel):
            reasoning: str = Field(description="Reasons for your proposal")
            evidence: str = Field(description="Evidence from the data")
        
        highest_arg = self.llm_engine(prompt=prompt_proposal, response_format=ProposalArgument)
        arg1_json = highest_arg.model_dump_json(indent=2)
        
        # Step 2: Opposition responds with final score
        prompt_opposition = self._create_opposition_prompt(
            other_modalities, highest_mod_name, highest_score, 
            arg1_json, evidence_dict, year
        )
        
        lowest_score = min([s for _, s in other_modalities])
        class DebateArgument(BaseModel):
            score: float = Field(description=f"Final risk score between {lowest_score:.4f} and {highest_score:.4f}")
            reason: str = Field(description="Reason and analysis for the predicted risk score")
        
        judgment = self.llm_engine(prompt=prompt_opposition, response_format=DebateArgument)
        return {
            'final_score': judgment.score,
            'method': 'debate_ensemble',
            'reasoning': judgment.reason,
            'debate_transcript': {
                'highest_argument': arg1_json,
                'judgment': judgment.model_dump()
            }
        }

    def _create_debate_prompt_highest(self, highest_modality: str, highest_score: float, 
                                     other_modalities: List[tuple], evidence: Dict[str, str]) -> str:
        """Prompt for highest-scoring modality to argue for elevated risk."""
        other_info = "\n".join([f"- {mod}: risk score {score:.4f}" for mod, score in other_modalities])
        all_modalities = [highest_modality] + [mod for mod, _ in other_modalities]
    
        return f"""You are the {highest_modality} modality defending your high risk score in a debate.

    Available modalities in this case: {', '.join(all_modalities)}
    Other modalities' scores:
    {other_info}

    Evidence you have access to: {evidence.get(highest_modality, 'Not provided')}

    Given that dementia is rare (only 3-5% of patients), you need to make a STRONG case if you believe this patient has higher dementia risk.

    The evidence you should consider including but not limited to:
    - memory deficit or memory and cognitive functions related symptoms
    - risk factors for dementia such as hypertension, diabetes, smoking, etc
    - MRI findings of brain atrophy
    - other symptoms that are suggestive of dementia (e.g. AD, Vascular, LBD, etc)

    Your task:
    1. Present the evidences that supports your high risk score
    2. Explain why your higher risk score should be taken seriously

    Provide your argument."""

    def _create_opposition_prompt(self, opposing_modalities: List[tuple], highest_modality: str, 
                                 highest_score: float, highest_argument: str, 
                                 evidence: Dict[str, str], year: int) -> str:
        """Prompt for opposing modalities to make final decision."""
        lowest_score = min([s for _, s in opposing_modalities])
        average_score = np.mean([s for _, s in opposing_modalities] + [highest_score])
        # medium_score = np.median([s for _, s in opposing_modalities] + [highest_score])
        
        opposing_mod_names = [mod for mod, _ in opposing_modalities]
        all_modalities = [highest_modality] + opposing_mod_names
        
        opposition_info = "\n".join([
            f"- {mod}: Risk score {score:.4f}, Evidence: {evidence.get(mod, 'Not provided')}"
            for mod, score in opposing_modalities
        ])
        dementia_criteria = self.obtain_dementia_criteria()
        return f"""You represent the opposing modalities ({', '.join(opposing_mod_names)}) defending your lower risk scores for dementia risk in a debate.

    Available modalities in this case: {', '.join(all_modalities)}

    Dementia criteria:
    {dementia_criteria}

    Argument from the {highest_modality} modality (risk score: {highest_score:.4f}):
    {highest_argument}

    Your assessments from the opposing modalities: 
    {opposition_info}

    Now combine the evidence you have and the argument from the {highest_modality} modality to make a final decision on the risk score.

    If the following evidences exist, it is strong evidence for increasing the risk in the next {year} years:
        - direct mention of dementia or related diseases in the notes or EHR
        - direct mention of memory deficit or memory and cognitive functions related symptoms in the notes or EHR
        - other symptoms that are suggestive of dementia (e.g. AD, Vascular, LBD, etc)
        - Brain atrophy that is typically affected by dementia and further confirmed by the volumes
        - other evidences that are as strong or direct as the above. 
    
    If you think the evidence is not enough to make a decision, you can assign a risk score be the average risk score {average_score:.4f}.
    The absolute value of this risk does not matter (even 0.1 can be high risk, 0.3 can be low risk depending on the risk range).
    
    Your task:
    1. Reasoning and analyzing all evidences to achieve the risk score within {year} years, be specific and detailed.
    2. Based on the analysis, propose a risk score, which must be in the range of low risk: {lowest_score:.4f} and high risk: {highest_score:.4f}. 
    """
    # def _create_opposition_prompt(self, opposing_modalities: List[tuple], highest_modality: str, 
    #                              highest_score: float, highest_argument: str, 
    #                              evidence: Dict[str, str], year: int) -> str:
    #     """Prompt for opposing modalities to make final decision."""
    #     lowest_score = min([s for _, s in opposing_modalities])
    #     average_score = np.mean([s for _, s in opposing_modalities] + [highest_score])
    #     # medium_score = np.median([s for _, s in opposing_modalities] + [highest_score])
        
    #     opposition_info = "\n".join([
    #         f"- {mod}: Evidence: {evidence.get(mod, 'Not provided')}"
    #         for mod, s in opposing_modalities
    #     ])
    #     dementia_criteria = self.obtain_dementia_criteria()
    #     return f"""You represent the other modalities ({', '.join([m for m, _ in opposing_modalities])}) analyzing this patient's dementia risk within the next {year} years. 

    # The {highest_modality} modality has proposed for higher dementia risk than you do:

    # --- THEIR ARGUMENT ---
    # {highest_argument}
    # --- END ARGUMENT ---

    # YOUR ASSESSMENTS:
    # {opposition_info}


    # The risk score should be calibrated to the range of: risk score of {highest_score:.4f} means the risk is high, and the risk score of {lowest_score:.4f} means the risk is low. 

    # Your task:
    # 1. Reasoning and analyzing all evidences to update the risk score within {year} years, be specific and detailed.
    # 2. Based on the analysis, propose a risk score, which must be in the range of low risk: {lowest_score:.4f} and high risk: {highest_score:.4f}. Provide your reasoning and the proposed risk score.
    # """
    
    def ensemble_methods(self, **modality_scores):
        """
        Flexible ensemble methods that work with >= 1 modalities.
        
        Args:
            **modality_scores: Variable keyword arguments for modality scores
                               (e.g., note_p=0.7, ehr_p=0.6, image_p=0.5)
        
        Returns:
            dict: Ensemble results
        """
        results = {}
        
        # Filter out None values and convert strings to floats
        valid_scores = {}
        for modality, score in modality_scores.items():
            if score is not None:
                if isinstance(score, str):
                    score = float(score)
                valid_scores[modality] = score
        
        if len(valid_scores) == 0:
            return {"error": "No valid modality scores provided"}
        
        scores_list = list(valid_scores.values())
        num_modalities = len(scores_list)
        scores_array = np.array(scores_list)
        
        # Handle single modality case
        if num_modalities == 1:
            single_score = scores_list[0]
            results['Average'] = single_score
            results['Maximum'] = single_score
            results['Minimum'] = single_score
            results['Majority'] = 1.0 if single_score > 0.5 else 0.0
            results['modalities_used'] = list(valid_scores.keys())
            results['num_modalities'] = 1
            results['single_modality'] = True
            return results
        
        # Handle multiple modalities (>= 2)
        avg = np.mean(scores_array)
        results['Average'] = avg
        
        maximum = np.max(scores_array)
        results['Maximum'] = maximum
        
        minimum = np.min(scores_array)
        results['Minimum'] = minimum
        
        votes = np.sum(scores_array > 0.5)
        majority = votes / num_modalities
        results['Majority'] = majority
        
        results['modalities_used'] = list(valid_scores.keys())
        results['num_modalities'] = num_modalities
        results['single_modality'] = False
        
        return results

    def execute(self, metadata_info: Dict[str, Any], task: str = "Predict patient risk", 
                extracted_evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute simple debate for collaborative synthesis across modalities.
        
        Args:
            metadata_info: Metadata information from multiple agents
            task: The prediction task description
            extracted_evidence: Optional dict with 'agent_evidence' and 'raw_evidence' for modalities
            
        Returns:
            Dictionary containing debate results
        """
        
        if not self.llm_engine:
            raise ValueError("LLM engine not initialized. Please provide a valid model_string when initializing the tool.")
        
        # try:
        # Extract agent data
        agent_data = self._extract_agent_data(metadata_info)
        year = metadata_info["dataset"].get("year", 3)
        
        if not agent_data:
            return {
                "status": "error",
                "message": "No valid agent data found in metadata",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get risk scores (only for available modalities)
        modality_scores_raw = {}
        all_modalities = ['image_agent', 'ehr_agent', 'note_agent']
        available_modalities = []
        missing_modalities = []
        
        for agent_key in all_modalities:
            modality_name = agent_key.replace('_agent', '')
            if agent_key in agent_data:
                score = agent_data[agent_key].get('risk_score')
                if score is not None:
                    modality_scores_raw[modality_name] = score
                    available_modalities.append(modality_name)
                else:
                    missing_modalities.append(modality_name)
            else:
                missing_modalities.append(modality_name)
        
        if missing_modalities:
            print(f"⚠️ Missing modalities: {', '.join(missing_modalities)}")
        print(f"✅ Available modalities: {', '.join(available_modalities)}")

        # Run simple debate
        debate_result = self._run_debate_ensemble(
            modality_scores_raw, 
            agent_data, extracted_evidence, year, 
            min_threshold=0.025
        )

        # Simple ensemble for comparison (with _p suffix for ensemble_methods)
        modality_scores = {
            f'{k}_p': v for k, v in modality_scores_raw.items()
        }
        simple_ensemble = self.ensemble_methods(**modality_scores)
        
        # Return simple result
        return {
            "status": "success",
            "task": task,
            "modality_scores": modality_scores,
            "available_modalities": available_modalities,
            "missing_modalities": missing_modalities,
            "final_risk_score": debate_result['final_score'],
            "reasoning": debate_result['reasoning'],
            "method": debate_result['method'],
            "debate_transcript": debate_result.get('debate_transcript'),
            "simple_ensemble": simple_ensemble,
            "timestamp": datetime.now().isoformat()
        }
            
        # except Exception as e:
        #     return {
        #         "status": "error",
        #         "message": f"Debate failed: {str(e)}",
        #         "timestamp": datetime.now().isoformat()
        #     }
if __name__ == "__main__":
    print("Multi-Agent Debate Tool - Simple debate approach")
    print("Use execute() method to run debate ensemble")
