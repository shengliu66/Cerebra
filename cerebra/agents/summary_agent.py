
from cerebra.agents.lightweight_agent import LightweightAgent
from typing import Dict, Any, List
from cerebra.utils.metadata import Metadata
import json
import os
import pickle
from datetime import datetime
from cerebra.utils.utils import create_montage_from_file
from pydantic import BaseModel, Field
import dotenv

# Load environment variables
dotenv.load_dotenv()
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")

class ModalityEvidenceExtraction(BaseModel):
    """Structured evidence extraction from modality data"""
    positive_evidence: List[str] = Field(description="Evidence (e.g. memory loss, confusion, disease like diabetes, syphilis, etc.) directly supporting elevated dementia risk (be specific)")
    negative_evidence: List[str] = Field(description="Evidence (e.g. cognitive normal, alert, etc.) directly supporting lower dementia risk (be specific)")
    key_findings: str = Field(description="Summary of the most relevant findings")


class SummaryAgent(LightweightAgent):
    """
    Summary agent that aggregates outputs from multiple modality agents
    and produces final predictions and recommendations using multi-agent debate.
    """
    
    def __init__(self, llm_engine_name: str = "gpt-4o", use_raw_evidence: bool = False, task: str = "classification_prediction"):
        super().__init__(
            agent_name="summary_agent",
            llm_engine_name=llm_engine_name,
            use_nyu_hipaa=True,
            enabled_tools=[],  # No specific tools needed
            verbose=True
        )
        
        self.use_raw_evidence = use_raw_evidence  # Option to extract evidence from raw data
        self.task = task  # Task type (e.g., "classification_prediction")
        # Initialize multimodal LLM for image analysis
        from cerebra.engine.factory import create_llm_engine
        self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=True, use_nyu_hipaa=True)
        
        # Initialize multi-agent debate tool
        from cerebra.tools.summary_agent.multi_agent_debate.tool import Multi_Agent_Debate_Tool
        self.debate_tool = Multi_Agent_Debate_Tool()
        self.debate_tool.set_llm_engine(model_string=llm_engine_name, is_multimodal=False, use_nyu_hipaa=True)
    
    def _get_data_path_suffix(self, year: int, modality: str) -> str:
        """Generate path suffix based on year, task, and modality.
        
        Args:
            year: Prediction year
            modality: One of 'ehr', 'note', 'image'
        
        Returns:
            Path suffix like 'NYU_3yr_survival_False_diagnosis_False_volume_False'
        """
        # Determine volume parameter based on modality
        volume = "True" if modality == "image" else "False"
        
        # Determine survival and diagnosis based on task
        if self.task == "classification_prediction":
            survival = "False"
            diagnosis = "False"
        else:
            # Add other task types here as needed
            survival = "False"
            diagnosis = "False"
        
        return f"NYU_{year}yr_survival_{survival}_diagnosis_{diagnosis}_volume_{volume}"
    
    def _get_data_agent_base_path(self, modality: str, year: int) -> str:
        """Get base path for data agent raw data.
        
        Args:
            modality: One of 'ehr', 'note', 'image'
            year: Prediction year
        
        Returns:
            Full base path to raw data directory
        """
        suffix = self._get_data_path_suffix(year, modality)
        return os.path.join(CACHE_DIR, "data_agent", f"{modality}_agent", suffix)
    
    def _get_summary_results_path(self, patient_id: int, year: int) -> str:
        """Get path to summary agent classification prediction results.
        
        Args:
            patient_id: Patient ID
            year: Prediction year
        
        Returns:
            Full path to JSON results file
        """
        filename = f"NYU_patient_{patient_id}_year_{year}_test_{self.task}_results.json"
        return os.path.join(CACHE_DIR, "summary_agent", self.task, filename)
    
    def obtain_dementia_criteria(self) -> str:
        """Obtain dementia criteria from the literature."""
        return """
        AD / ADRD (Alzheimer’s disease and related dementias)
        Includes diagnoses such as: Vascular dementia, Dementia due to other underlying diseases, with or without behavioral disturbance, Unspecified dementia, Amnestic disorders caused by known physiological conditions, Progressive supranuclear palsy, Alzheimer’s disease, Frontotemporal dementias (including Pick’s disease and other variants), Dementia with Lewy bodies, Degenerative diseases of the nervous system, Age-related (senile) degeneration of the brain

        Mild Cognitive Impairment (MCI)
        Includes conditions such as: Mild cognitive impairment of uncertain or unknown cause, Corticobasal degeneration

        Dementia-Related Medications: Common medications used in the treatment of dementia symptoms, including: Donepezil, Galantamine, Memantine, Rivastigmine, Tacrine
        """
    
    def _extract_evidence_from_raw_data(self, patient_id: int, year: int, metadata_info: Dict[str, Any]) -> tuple:
        """Extract evidence from raw patient data using LLM analysis.
        
        Returns:
            (evidence_dict, raw_data_dict): Evidence for debate and raw data for chat package
        """
        evidence = {}
        raw_data_cache = {'demographic': None, 'ehr_raw': None, 'note_raw': None}
        
        # Extract EHR evidence
        try:
            base_path = self._get_data_agent_base_path('ehr', year)
            raw_data_path = os.path.join(base_path, f"NYU_{patient_id}_raw_data.pkl")
            demographic_path = os.path.join(base_path, f"NYU_{patient_id}_demographic_info.pkl")
            dementia_criteria = self.obtain_dementia_criteria()

            ehr_data = {}
            if os.path.exists(raw_data_path):
                with open(raw_data_path, 'rb') as f:
                    ehr_data['raw_data'] = pickle.load(f)
                    raw_data_cache['ehr_raw'] = ehr_data['raw_data']  # Cache for reuse
            if os.path.exists(demographic_path):
                with open(demographic_path, 'rb') as f:
                    ehr_data['demographic'] = pickle.load(f)
                    raw_data_cache['demographic'] = ehr_data['demographic']  # Cache for reuse
            
            if ehr_data:
                
                data_summary = f"Patient ID: {patient_id}\n\n"
                if 'demographic' in ehr_data:
                    data_summary += f"DEMOGRAPHIC INFO:\n{ehr_data['demographic']}\n\n"
                if 'raw_data' in ehr_data:
                    raw_str = str(ehr_data['raw_data'])
                    if len(raw_str) > 10000:
                        raw_str = raw_str[:10000] + "... (truncated)"
                    data_summary += f"EHR RAW DATA:\n{raw_str}\n"
                
                prompt = f"""Analyze the following EHR (Electronic Health Record) data for dementia risk assessment.
            DISEASE CRITERIA:
            {dementia_criteria}

            {data_summary}

            Your task:
            1. Identify specific findings that could indicate dementia risk (positive evidence) 
            2. Identify specific findings that suggest normal cognitive status (negative evidence)
            3. Provide the original EHR data that supports the evidence
  
            If the positive evidence and negative evidence is directly contradictory, then only include the positive evidence.

            Be specific and reference actual values from the data."""
                
                result = self.llm_engine(prompt=prompt, response_format=ModalityEvidenceExtraction)
                evidence['ehr'] = {
                    'positive_evidence': result.positive_evidence,
                    'negative_evidence': result.negative_evidence,
                    'key_findings': result.key_findings
                }
        except Exception as e:
            print(f"⚠️ Could not extract EHR evidence: {e}")
            evidence['ehr'] = None
        
        # Extract Note evidence
        try:
            base_path = self._get_data_agent_base_path('note', year)
            raw_data_path = os.path.join(base_path, f"NYU_{patient_id}_raw_data.pkl")
            
            note_data = {}
            if os.path.exists(raw_data_path):
                with open(raw_data_path, 'rb') as f:
                    note_data['raw_data'] = pickle.load(f)
                    raw_data_cache['note_raw'] = note_data['raw_data']  # Cache for reuse
            
            if note_data:
                data_summary = f"Patient ID: {patient_id}\n\n"
    
                if 'demographic' in note_data:
                    data_summary += f"DEMOGRAPHIC INFO:\n{note_data['demographic']}\n\n"
                
                if 'raw_data' in note_data:
                    # Convert raw data to string representation (truncate if too long)
                    for i in range(len(note_data['raw_data']['indexed_notes'])):
                        raw_str = str(note_data['raw_data']['indexed_notes'][::-1][i]['full_text'])
                        if len(raw_str) > 10000:
                            raw_str = raw_str[:10000] + "... (truncated)"
                            break
                    data_summary += f"CLINICAL NOTES:\n{raw_str}\n"
                
                prompt = f"""Analyze the following clinical notes for dementia risk assessment.
            DISEASE CRITERIA:
            {dementia_criteria}

            {data_summary}

            Your task:
            1. Identify specific mentions that could indicate higher dementia risk in the future. (positive evidence) 
            2. Identify specific mentions that could indicate lower dementia risk in the future. (negative evidence)
            3. Be aware the date of the notes, only include the most recent evidence.
            Be specific and quote relevant phrases from the notes. Be as comprehensive as possible."""
                
                result = self.llm_engine(prompt=prompt, response_format=ModalityEvidenceExtraction)
                evidence['note'] = {
                    'positive_evidence': result.positive_evidence,
                    'negative_evidence': result.negative_evidence,
                    'key_findings': result.key_findings
                }
        except Exception as e:
            print(f"⚠️ Could not extract note evidence: {e}")
            evidence['note'] = None
        
        # Extract Image evidence
        try:
            json_path = self._get_summary_results_path(patient_id, year)
            image_path = None
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if 'image_agent' in data and 'mri_visualizations' in data['image_agent']['dataset']:
                    mri_viz = data['image_agent']['dataset']['mri_visualizations']
                    if 'configuration' in mri_viz and 'value' in mri_viz['configuration']:
                        viz_paths = mri_viz['configuration']['value']
                        if isinstance(viz_paths, list) and len(viz_paths) > 0:
                            image_path = viz_paths[0]
                if 'image_agent' in data and 'evidence' in data['image_agent']['dataset']:
                    image_volumes = data['image_agent']['dataset']['evidence']['configuration']['value'][0]
                    image_volumes = [f"Brain region: {e['feature']} with volume {e['feature_value']:.8f}, Average volume for the healthy group (for reference only): {e['average_feature_volume']:.8f}" for e in image_volumes]
                    image_volumes_str = "\n".join(image_volumes)
                else:
                    image_volumes_str = "No image volumes available"
            
            if image_path and os.path.exists(image_path):
                prompt = f"""Analyze this brain MRI scan for signs of dementia or cognitive impairment.
            
            DISEASE CRITERIA:
            {dementia_criteria}

            Identify:
            1. Specific findings indicating brain atrophy, ventricular enlargement, white matter hyperintensities, etc. (positive evidence)
            2. Brain region size changes that first observed from the image and then confirmed by the volumes (positive evidence)
            3. Direct findings suggesting normal brain structure (negative evidence)

            RULES:
             - Brain volumes that are slightly decreased or increased compared to healthy average are not considered as evidence because individual variance is expected.
             - Here are the volumes for the brain regions obtained from segmentation (for reference only):
                {image_volumes_str}

            You analysis should be based on the scan, with volumes as reference, identify the positive and negative evidence indicating brain structure changes or abnormalities.

            Be specific about what you observe in the images."""
                
                image_bytes = create_montage_from_file(image_path)
                content = [prompt, image_bytes]
                
                # Use multimodal LLM from debate tool
                if self.llm_engine_mm:
                    result = self.llm_engine_mm(content, response_format=ModalityEvidenceExtraction)
                    evidence['image'] = {
                        'positive_evidence': result.positive_evidence,
                        'negative_evidence': result.negative_evidence,
                        'key_findings': result.key_findings
                    }
        except Exception as e:
            print(f"⚠️ Could not extract image evidence: {e}")
            evidence['image'] = None
        
        return evidence, raw_data_cache  # Return both evidence and cached raw data

    def _extract_agent_evidence(self, metadata_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract original evidence from agent outputs."""
        evidence = {}
        dataset = metadata_info.get('dataset', {})
        
        # EHR agent evidence
        ehr_outputs = dataset.get('ehr_agent_outputs', {}).get('dataset', {})
        ehr_evidence_original = ehr_outputs.get('evidence', {}).get('configuration', {}).get('value', [[]])[0]
        ehr_evidence = [e.split('Feature importance: ')[0].replace('Feature in EHR: ', '') for e in ehr_evidence_original]
        evidence['ehr'] = ehr_evidence if ehr_evidence else None
        
        # Note agent evidence
        note_outputs = dataset.get('note_agent_outputs', {}).get('dataset', {})
        note_evidence_original = note_outputs.get('evidence', {}).get('configuration', {}).get('value', [[]])[0]
        note_evidence = [e['text'] for e in note_evidence_original]
        evidence['note'] = note_evidence if note_evidence else None
        
        # Image agent - get MRI visualization paths
        image_outputs = dataset.get('image_agent_outputs', {}).get('dataset', {})
        image_evidence_original = image_outputs.get('evidence', {}).get('configuration', {}).get('value', [[]])[0]
        mri_viz = image_outputs.get('mri_visualizations', {}).get('configuration', {}).get('value', [])
        mri_viz_three_axis = image_outputs.get('mri_visualizations_three_axis', {}).get('configuration', {}).get('value', [])
        image_evidence = [f"Brain region: {e['feature']} with volume {e['feature_value']:.8f}, Average volume for the healthy group (for reference only): {e['average_feature_volume']:.8f}" for e in image_evidence_original]
        evidence['image'] = {'volume': image_evidence, 'mri': mri_viz, 'mri_three_axis': mri_viz_three_axis} if mri_viz else {'volume': image_evidence, 'mri_segments': None, 'mri_three_axis': None} 
        return evidence

    def _extract_agent_value(self, agent_output, field_name, default=None):
        """
        Helper method to safely extract values from nested agent output structure.
        
        Args:
            agent_output: Agent output dictionary (could be nested)
            field_name: Field to extract (e.g., 'prediction', 'risk_score', 'evidence')
            default: Default value if field not found
        
        Returns:
            Extracted value or default
        """
        if not agent_output:
            return default
        
        # Try nested format: agent_output['dataset'][field_name]['configuration']['value']
        if 'dataset' in agent_output and isinstance(agent_output['dataset'], dict):
            dataset = agent_output['dataset']
            if field_name in dataset:
                field_data = dataset[field_name]
                if isinstance(field_data, dict):
                    if 'configuration' in field_data and 'value' in field_data['configuration']:
                        value = field_data['configuration']['value']
                        # If it's a list with one element, extract it
                        if isinstance(value, list) and len(value) > 0:
                            return value[0] if field_name in ['prediction', 'risk_score'] else value
                        return value
        
        # Try direct access: agent_output[field_name]
        if field_name in agent_output:
            return agent_output[field_name]
        
        return default
        

    def run(self, task: str, input_metadata: Metadata) -> Metadata:
        """
        Use multi-agent debate to analyze agent outputs and generate final prediction and recommendations.
        """
        # Get metadata info and year for output
        metadata_info = input_metadata.get_metadata_info()
        year = metadata_info["dataset"]["year"] if "year" in metadata_info["dataset"] else 1
        patient_id = metadata_info["dataset"]["patient_id"]
        institution = metadata_info["dataset"]["institution"]
        
        # Use multi-agent debate tool for analysis 
        final_result = self._multi_agent_debate_analysis(task, metadata_info)
        print('✅ Analysis complete')
        
        chat_package = self._create_chat_package(patient_id, year, metadata_info, final_result)
        final_result["chat_package"] = chat_package
        print('✅ Chat package created')
        # Save and return as Metadata
        return self._create_output(final_result, task, patient_id, year, institution)

    def chat(self, patient_id: str, year: int, query: str, 
             chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Interactive chat about patient using chat package data.
        Uses RAG approach - LLM retrieves relevant info from chat package.
        
        Args:
            patient_id: Patient ID
            year: Prediction year
            query: User's question
            chat_history: Previous conversation (optional, list of {'query': str, 'answer': str})
            
        Returns:
            Dict with answer, citations, confidence, follow_up_suggestions, and optional highlighted_evidence
            
        Example:
            >>> agent = SummaryAgent()
            >>> response = agent.chat(patient_id="0", year=1, query="Why is the risk score low?")
            >>> print(response['answer'])
            >>> print(response['citations'])
        """
        from cerebra.agents.utils.chat_utils import PatientChatHandler
        
        # Initialize chat handler with LLM engine
        chat_handler = PatientChatHandler(self.llm_engine)
        
        # Process query using RAG
        return chat_handler.process_query(patient_id, year, query, chat_history)

    def _multi_agent_debate_analysis(self, task: str, metadata_info: Dict[str, Any]) -> Dict[str, Any]:
        """Call debate tool to analyze agent outputs (following dissertation approach)."""
        try:
            # Get patient_id and year
            patient_id = metadata_info.get("dataset", {}).get("patient_id", 0)
            year = metadata_info.get("dataset", {}).get("year", 3)
            
            # Extract evidence (3 options: agent evidence, raw evidence, or both)
            agent_evidence = self._extract_agent_evidence(metadata_info)
            extracted_evidence = None
            raw_data_cache = None
            
            if self.use_raw_evidence:
                print("🔍 Extracting evidence from raw patient data...")
                extracted_evidence, raw_data_cache = self._extract_evidence_from_raw_data(int(patient_id), year, metadata_info)
                print("✅ Evidence extraction complete")
            
            # Combine evidence if both available
            combined_evidence = {
                'agent_evidence': agent_evidence,
                'raw_evidence': extracted_evidence,
            }

            # Call the debate tool
            debate_result = self.debate_tool.execute(metadata_info, task, extracted_evidence=combined_evidence)
            
            if debate_result.get("status") == "success":
                result = {
                    "modality_scores": debate_result.get("modality_scores", {}),
                    "aggregated_risk_score": debate_result.get("final_risk_score", 0.5),
                    "final_analysis": debate_result.get("reasoning", "No analysis"),
                    "method": debate_result.get("method", "debate_ensemble"),
                    "debate_transcript": debate_result.get("debate_transcript"),
                    "simple_ensemble": debate_result.get("simple_ensemble", {}),
                    "extracted_evidence": combined_evidence,
                    "raw_data_cache": raw_data_cache,  # Pass cached raw data to avoid reloading
                    "timestamp": debate_result.get("timestamp", datetime.now().isoformat())
                }
                return result
            else:
                return {
                    "error": debate_result.get("message"),
                    "modality_scores": debate_result.get("modality_scores", {}),
                    "aggregated_risk_score": 0.5,
                    "final_analysis": "Debate failed",
                    "extracted_evidence": combined_evidence,
                    "raw_data_cache": raw_data_cache,
                    "timestamp": datetime.now().isoformat(),
                    "method": "Error"
                }
            
        except Exception as e:
            print(f"⚠️ Error in debate analysis: {e}")
            return {
                "error": str(e),
                "aggregated_risk_score": 0.5,
                "final_analysis": "Debate analysis failed due to error",
                "extracted_evidence": {},
                "raw_data_cache": None,
                "timestamp": datetime.now().isoformat(),
                "method": "Error"
            }


    def _create_chat_package(self, patient_id: str, year: int, metadata_info: Dict[str, Any], 
                              analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive patient data package for chat backend."""
        import pickle
        from cerebra.agents.data_agent import DataAgent
        
        # Check if we have cached raw data from debate phase
        raw_data_cache = analysis_result.get('raw_data_cache')
        
        # Get demographic info
        demographic_info = None
        
        if raw_data_cache and raw_data_cache.get('demographic'):
            # Reuse cached data from debate phase
            demographic_info = raw_data_cache['demographic']
            print(f"✅ Reused cached demographic_info for patient {patient_id}")
        else:
            # Load via DataAgent if not cached
            try:
                data_agent = DataAgent()
                data_agent.institution = metadata_info["dataset"]["institution"]
                data_agent.year = metadata_info["dataset"]["year"]
                demographic_info = data_agent.get_demographic_info(int(patient_id), year=year)
                print(f"✅ Loaded demographic_info for patient {patient_id}")
            except Exception as e:
                print(f"⚠️ Could not load demographic_info: {e}")
        
        # Get raw notes data
        note_data = None
        if raw_data_cache and raw_data_cache.get('note_raw'):
            # Reuse cached data and process with DataAgent
            try:
                note_data = raw_data_cache['note_raw']
                print(f"✅ Reused cached note raw_data for patient {patient_id}")
            except Exception as e:
                print(f"⚠️ Could not process cached note data: {e}")
                note_data = None
        
        if note_data is None and 'note_agent_outputs' in metadata_info['dataset']:
            # Fallback: load from metadata if not cached
            note_dataset = metadata_info['dataset']['note_agent_outputs']
            if 'test_data' in note_dataset:
                test_data_path = note_dataset['test_data'].get('saved_path')
                try:
                    data_agent = DataAgent()
                    with open(test_data_path, 'rb') as f:
                        test_data = pickle.load(f)
                    if isinstance(test_data, list) and len(test_data) > 0:
                        note_data = data_agent.get_raw_notes_with_indices(test_data[0])
                    else:
                        note_data = data_agent.get_raw_notes_with_indices(test_data)
                    print(f"✅ Loaded note raw_data for patient {patient_id}")
                except Exception as e:
                    print(f"⚠️ Could not load note raw_data: {e}")
        
        # Get EHR raw features
        ehr_raw_data = None
        if raw_data_cache and raw_data_cache.get('ehr_raw'):
            # Reuse cached data and process with DataAgent
            try:
                ehr_raw_data = raw_data_cache['ehr_raw']
                print(f"✅ Reused cached EHR raw_data for patient {patient_id}")
            except Exception as e:
                print(f"⚠️ Could not process cached EHR data: {e}")
                ehr_raw_data = None
        
        if ehr_raw_data is None and 'ehr_agent_outputs' in metadata_info['dataset']:
            # Fallback: load from metadata if not cached
            ehr_dataset = metadata_info['dataset']['ehr_agent_outputs']
            if 'test_data' in ehr_dataset:
                test_data_path = ehr_dataset['test_data'].get('saved_path')
                try:
                    data_agent = DataAgent()
                    with open(test_data_path, 'rb') as f:
                        test_data = pickle.load(f)
                    import pandas as pd
                    if isinstance(test_data, pd.DataFrame):
                        ehr_raw_data = data_agent.get_ehr_nonzero_features(test_data.iloc[0])
                    elif isinstance(test_data, list) and len(test_data) > 0:
                        ehr_raw_data = data_agent.get_ehr_nonzero_features(test_data[0])
                    else:
                        ehr_raw_data = data_agent.get_ehr_nonzero_features(test_data)
                    print(f"✅ Loaded EHR raw_data for patient {patient_id}")
                except Exception as e:
                    print(f"⚠️ Could not load EHR raw_data: {e}")
        
        # Extract only serializable modality outputs (avoid circular references)
        def extract_serializable_data(agent_outputs):
            """Extract only the essential, serializable data from agent outputs."""
            if not agent_outputs:
                return {}
            
            serializable = {}
            for key, value in agent_outputs.items():
                if isinstance(value, dict):
                    # Only include 'configuration' which has the actual values
                    if 'configuration' in value:
                        serializable[key] = value['configuration']
                    elif 'saved_path' in value or 'description' in value:
                        # Keep metadata but not nested objects
                        serializable[key] = {
                            'saved_path': value.get('saved_path'),
                            'description': value.get('description')
                        }
            return serializable
        
        # Extract modality outputs without circular references
        modality_outputs = {
            'note_agent': extract_serializable_data(metadata_info['dataset'].get('note_agent_outputs', {})),
            'image_agent': extract_serializable_data(metadata_info['dataset'].get('image_agent_outputs', {})),
            'ehr_agent': extract_serializable_data(metadata_info['dataset'].get('ehr_agent_outputs', {}))
        }
        
        # Get extracted evidence from analysis result
        extracted_evidence = analysis_result.get('extracted_evidence', {})
        
        # Extract only essential analysis data (avoid including full objects)
        summary_analysis = {
            'probabilities_from_trained_models': analysis_result.get('modality_scores', {}),
            'aggregated_risk_score': analysis_result.get('aggregated_risk_score', 0.5),
            'final_analysis': analysis_result.get('final_analysis', ''),
            'debate_history': analysis_result.get('debate_transcript', ''),
        }
        
        # Build chat package with only serializable data
        chat_package = {
            'patient_id': patient_id,
            'year': year,
            'demographics': demographic_info,
            'raw_notes_indexed': note_data,
            'ehr_raw_features': ehr_raw_data,
            'modality_outputs': modality_outputs,
            'agent_evidence': extracted_evidence.get('agent_evidence', {}),
            'extracted_evidence_from_raw_data': extracted_evidence.get('raw_evidence', {}),
            'summary_analysis': summary_analysis,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_notes': note_data.get('total_notes', 0) if note_data else 0,
                'has_ehr_data': ehr_raw_data is not None,
                'has_imaging_data': 'image_agent_outputs' in metadata_info['dataset']
            }
        }
        
        return chat_package

    def _create_output(self, result: Dict[str, Any], task: str, patient_id: str, year: int, institution: str='NYU') -> dict:
        """Create Metadata object from analysis results."""
        # Save result to file
        cache_dir = os.path.join(CACHE_DIR, self.agent_name, self.task)
        os.makedirs(cache_dir, exist_ok=True)
        result_file = os.path.join(cache_dir, f"summary_results_{institution}_patient_{patient_id}_{year}year.json")
        # Save chat package for dashboard (do this first, before removing from result)
        chat_package = result.get("chat_package", {})
        chat_package_file = None
        if chat_package:
            chat_package_file = os.path.join(cache_dir, f"chat_package_{institution}_patient_{patient_id}_{year}year.json")
            with open(chat_package_file, 'w') as f:
                json.dump(chat_package, f, indent=2, default=str)
        
        # Remove chat_package from result before saving
        result_copy = result.copy()
        result_copy.pop("chat_package", None)
        
        # Save JSON results
        with open(result_file, 'w') as f:
            json.dump(result_copy, f, indent=2, default=str)
        
        # Create dictionary using the correct signature
        result_dict = {
            "status": "success",
            "dataset": {
                "summary_results": result,
                "result_file_path": result_file,
                "chat_package_path": chat_package_file,
                "task": task
            },
            "final": {
                "summary_results": result,
                "result_file_path": result_file,
                "chat_package_path": chat_package_file,
                "task": task
            },
            "model": {},
            "cache_directory": cache_dir,
            "agent_name": self.agent_name,
        }
        return result_dict


    def register_agent_capabilities(self) -> Dict[str, Any]:
        """Register the capabilities of the summary agent."""
        return {
            "summary_agent": {
                "agent_description": "Summary agent for aggregating and analyzing multiple agent predictions using multi-agent debate",
                "agent_capabilities": [
                    "multi-agent collaborative debate across modalities",
                    "modality-specific expert analysis (EHR, clinical notes, imaging)",
                    "consensus-based final predictions and risk assessment", 
                    "producing personalized patient recommendations",
                    "creating structured assessment reports",
                    "transparent reasoning through expert debate synthesis",
                    "Used after the usage of the modality agents"
                ],
                "agent_input_types": {
                    "task": "str",
                    "input_metadata": "Metadata"
                },
                "agent_output_type": "Metadata object containing final analysis and recommendations"
            }
        }

if __name__ == "__main__":
    # Test with multi-agent debate
    agent = SummaryAgent()
    
    # Create merged metadata simulating what orchestrator would pass
    # This uses the exact format from the real system
    merged_agent_data = {
        "ehr_agent_outputs": {
            'train_data': {'saved_path': 'cerebra_cache/data_agent/metadata/train_data.pkl', 'description': 'EHR data for training, list of sparse matrices', 'configuration': {'num_samples': 1296}}, 
            'validation_data': {'saved_path': 'cerebra_cache/data_agent/metadata/validation_data.pkl', 'description': 'EHR data for validation, list of sparse matrices', 'configuration': {'num_samples': 325}}, 
            'test_data': {'saved_path': 'cerebra_cache/data_agent/metadata/test_data.pkl', 'description': 'EHR data for testing, list of sparse matrices', 'configuration': {'num_samples': 1}},
            'prediction': {'saved_path': None, 'description': 'Prediction results', 'configuration': {'value': [0], 'num_samples': 1}}, 
            'risk_score': {'saved_path': None, 'description': 'Risk scores for each prediction', 'configuration': {'value': [0.52], 'num_samples': 1}}, 
            'evidence': {'saved_path': None, 'description': 'Evidence sentences for each prediction', 'configuration': {'value': [['Feature Name in EHR: Diagnosis:Disorder of the nose sid:89488007 232340005, corresponding value: 2.0, Importance score: 0.0276. ', 'Feature Name in EHR: ProcedureCPT:chg hepatitis c ab test 86803, corresponding value: 3.0, Importance score: 0.0188. ', 'Feature Name in EHR: Diagnosis:Adverse effect of monoamine-oxidase-inhibitor antidepressants, sequela T43.1X5S, corresponding value: 2.0, Importance score: 0.0178. ', 'Feature Name in EHR: Diagnosis:Irritable bowel syndrome with diarrhea K58.0, corresponding value: 2.0, Importance score: 0.0176. ', 'Feature Name in EHR: Diagnosis:Secondary malignant neoplasm of unspecified lung C78.00, corresponding value: 7.0, Importance score: 0.0168. ']], 'num_samples': 1}}
        },
        "note_agent_outputs": {
            'train_data': {'saved_path': 'cerebra_cache/data_agent/metadata/train_data.pkl', 'description': 'Note data for training', 'configuration': {'num_samples': 1296}}, 
            'validation_data': {'saved_path': 'cerebra_cache/data_agent/metadata/validation_data.pkl', 'description': 'Note data for validation', 'configuration': {'num_samples': 325}}, 
            'test_data': {'saved_path': 'cerebra_cache/data_agent/metadata/test_data.pkl', 'description': 'Note data for testing', 'configuration': {'num_samples': 1}},
            'prediction': {'saved_path': None, 'description': 'Prediction results', 'configuration': {'value': [1], 'num_samples': 1}}, 
            'risk_score': {'saved_path': None, 'description': 'Risk scores for each prediction', 'configuration': {'value': [0.78], 'num_samples': 1}}, 
            'evidence': {'saved_path': None, 'description': 'Evidence sentences for each prediction', 'configuration': {'value': [['High attention weight on sentence: "Patient reports increasing difficulty with memory and concentration."', 'High attention weight on sentence: "Family history significant for dementia in mother at age 72."', 'High attention weight on sentence: "Recent cognitive assessment shows mild impairment in executive function."']], 'num_samples': 1}}
        },
        "patient_id": "test_patient_123",
        "year": 1
    }
    
    # Create merged metadata object
    merged_metadata = Metadata.create_agent_output(
        status="success",
        dataset=merged_agent_data,
        model={},
        cache_directory="cerebra_cache/summary_agent",
        agent_name="merged_agents",
        status_description="Merged outputs from all agents"
    )
    
    # Test the summary agent
    print("Testing Summary Agent with multi-agent debate...")
    result = agent.run("Predict dementia risk for a patient", merged_metadata)
    
    # Display results
    print("✅ Summary agent completed successfully!")
    result_data = result if isinstance(result, dict) else result.get('dataset', {}).get('summary_results', {})
    print(f"🎯 Risk Score: {result_data.get('aggregated_risk_score', 'N/A')}")
    print(f"📝 Method: {result_data.get('method', 'N/A')}")
    
    print("\n--- Multi-Agent Debate Analysis Complete ---")
