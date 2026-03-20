import os
import pickle
import torch
import numpy as np
import torch.nn as nn
from cerebra.tools.base import BaseTool
from cerebra.utils.metadata import Metadata
from cerebra.tools.note_agent.models.sentence_attention import SentenceAttentionBERT
from cerebra.utils.log_utils import setup_logger
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")
# Setup logging
log_dir = os.path.join(CACHE_DIR, "note_agent", "logs")
logger = setup_logger(log_dir)

class Note_Model_Inference_Tool(BaseTool):
    def __init__(self):
        super().__init__()
        self.output_dir = os.path.join("cerebra_cache", "note_agent", "inference")
        os.makedirs(self.output_dir, exist_ok=True)

        self.set_metadata(
            tool_name="Note_Model_Inference_Tool",
            tool_description="Predict using trained note classification or survival model and return predictions with risk scores/survival curves and evidence.",
            tool_version="1.1.0",
            input_types={
                "test_features_path(required)": "str - Path to saved test features (.pkl)",
                "test_labels_path(required)": "str - Path to saved test labels (.npy)",
                "test_sentences_path(required)": "str - Path to test sentences (.pkl)",
                "trained_model_path(required)": "str - Path to the trained model (.pt)",
                "embedding_dim": "int - Dimension of input embeddings (default: 1024)",
                "top_k": "int - Number of top sentences to return as evidence (default: 15)",
                "task_type": "str - Type of task: 'classification' or 'survival' (default: 'classification')"
            },
            demo_commands=[
                {
                    "command": (
                        "    test_features_path=dataset['test_features']['saved_path'],\n"
                        "    test_labels_path=dataset['test_labels']['saved_path'],\n"
                        "    test_sentences_path=dataset['test_sentences']['saved_path'],\n"
                        "    trained_model_path=model['trained_model']['saved_path'],\n"
                        "    embedding_dim=model['trained_model']['configuration']['training_hyperparameters']['embedding_dim'],\n"
                        "    task_type='survival' if 'cox' in model['trained_model']['saved_path'] else 'classification',\n"
                        "    top_k=15\n"
                        ")"
                    ),
                    "description": "Predict using trained note model and return predictions/survival curves with evidence."
                }
            ],
            user_metadata={
                "limitations": [
                    "Only supports test features not original data",
                    "Survival predictions require task_type='survival'",
                ],
            },
            output_type="Metadata - Inference results with metrics, risk scores/survival curves, and evidence sentences",
            evaluation_criteria={
                "technical_success": "Does the tool execute without errors?",
                "inference_success": "Is the prediction for each note provided?",
                "evidence_quality": "Are the evidence sentences extracted for each prediction?",
                "risk_score": "Is the risk score or survival curve provided?"
            }
        )

    def compute_survival_curves(self, log_hazards, time_points):
        """
        Compute survival curves from log-hazards using exponential approximation.
        
        Args:
            log_hazards: Array of log-hazard predictions
            time_points: Array of time points at which to evaluate survival
            
        Returns:
            survival_probs: Survival probabilities at each time point
        """
        # Convert log-hazards to hazards
        hazards = np.exp(log_hazards)
        
        # For each time point, compute survival probability: S(t) = exp(-hazard * t)
        # This assumes a proportional hazards model with exponential baseline
        survival_probs = []
        for t in time_points:
            # Scale time appropriately (assuming time is in days, convert to years)
            survival_at_t = np.exp(-hazards * (t / 365.0))
            survival_probs.append(survival_at_t)
        
        return np.array(survival_probs).T  # Shape: (n_samples, n_timepoints)

    def compute_median_survival(self, log_hazards):
        """
        Compute median survival time from log-hazards.
        
        Args:
            log_hazards: Array of log-hazard predictions
            
        Returns:
            median_times: Median survival time for each sample (in years)
        """
        hazards = np.exp(log_hazards)
        # For exponential distribution: median = ln(2) / hazard
        # Convert from days to years
        median_times = (np.log(2) / hazards) / 365.0
        return median_times
    
    def compute_token_similarity(self, text1, text2):
        """Compute Jaccard similarity between two texts based on tokens"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if union else 0.0
    
    def deduplicate_evidence(self, candidate_evidence, similarity_threshold=0.85):
        """Remove duplicate/similar sentences, keeping ones with highest delta_risk"""
        deduplicated = []
        for item in candidate_evidence:
            is_duplicate = False
            for i, existing in enumerate(deduplicated):
                if self.compute_token_similarity(item['text'], existing['text']) > similarity_threshold:
                    is_duplicate = True
                    if abs(item['delta_risk_score']) > abs(existing['delta_risk_score']):
                        deduplicated[i] = item
                    break
            if not is_duplicate:
                deduplicated.append(item)
        return deduplicated

    def execute(self,
                test_features_path: str,
                test_labels_path: str,
                test_sentences_path: str,
                trained_model_path: str,
                embedding_dim: int = 1024,
                top_k: int = 15,
                task_type: str = "classification",
                **kwargs):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load inputs
            with open(test_features_path, 'rb') as f:
                test_embeddings = pickle.load(f)
            
            test_labels = np.load(test_labels_path, allow_pickle=True)
            with open(test_sentences_path, 'rb') as f:
                test_sentences = pickle.load(f)

            model = SentenceAttentionBERT(
                sentence_embed_dim=embedding_dim,
                weight_dim=embedding_dim,
                dropout=0,
                classifier_dropout=0
            ).to(device)
            model.load_state_dict(torch.load(trained_model_path, map_location=device))
            model.eval()

            # Auto-detect task type from model path if not specified
            if "cox" in trained_model_path.lower() or "survival" in trained_model_path.lower():
                task_type = "survival"
            
            is_survival = (task_type == "survival")
            
            if is_survival:
                logger.info("Running survival inference (Cox model)")
                log_hazards, all_weights, evidence = [], [], []
                
                for emb, label, sent in zip(test_embeddings, test_labels, test_sentences):
                    emb_array = np.array(emb)[:256,:]
                    x = torch.tensor(emb_array, dtype=torch.float32).unsqueeze(0).to(device)
                    x_full = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _, _ = model(x)  # logits are log-hazards for Cox
                        _, attn_weights, _ = model(x_full)
                    
                    log_hazard = logits.squeeze().item()
                    log_hazards.append(log_hazard)
                    all_weights.append(attn_weights.squeeze(0).cpu().tolist())
                    
                    # Compute baseline risk score
                    baseline_risk = 1.0 - np.exp(-np.exp(log_hazard) * (365.0 / 365.0))  # 1-year risk
                    
                    # Step 1: Identify high-attention sentences (attention > mean)
                    attn_np = attn_weights.squeeze(0).cpu().numpy()
                    mean_val = attn_np.mean()
                    high_attn_idx = np.where(attn_np > mean_val)[0]
                    
                    # Step 2: Perform ablation test on each high-attention sentence
                    candidate_evidence = []
                    for idx in high_attn_idx:
                        if idx < len(sent):
                            # Perform ablation test: remove this sentence and re-run prediction
                            emb_ablated = [emb[i] for i in range(len(emb)) if i != idx]
                            if len(emb_ablated) > 0:
                                emb_ablated_array = np.array(emb_ablated)
                                x_ablated = torch.tensor(emb_ablated_array, dtype=torch.float32).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    logits_ablated, _, _ = model(x_ablated)
                                log_hazard_ablated = logits_ablated.squeeze().item()
                                ablated_risk = 1.0 - np.exp(-np.exp(log_hazard_ablated) * (365.0 / 365.0))  # 1-year risk
                                delta_risk = baseline_risk - ablated_risk
                            else:
                                delta_risk = 0.0
                            
                            if isinstance(sent[idx], dict):
                                # New structured format with metadata
                                candidate_evidence.append({
                                    'text': sent[idx]['text'],
                                    'note_idx': sent[idx]['note_idx'],
                                    'sentence_idx': sent[idx]['sentence_idx'],
                                    'source_paragraph': sent[idx].get('source_paragraph', 0),
                                    'attention_weight': float(attn_weights.squeeze(0).cpu().numpy()[idx]),
                                    'delta_risk_score': float(delta_risk)
                                })
                            else:
                                # Backward compatibility: plain string
                                candidate_evidence.append({
                                    'text': sent[idx],
                                    'attention_weight': float(attn_weights.squeeze(0).cpu().numpy()[idx]),
                                    'delta_risk_score': float(delta_risk)
                                })
                    
                    # Step 2.5: Deduplicate similar sentences
                    candidate_evidence = self.deduplicate_evidence(candidate_evidence)
                    
                    # Step 3: Sort by delta_risk and select top k
                    candidate_evidence.sort(key=lambda x: abs(x['delta_risk_score']), reverse=True)
                    evidence_items = candidate_evidence[:top_k]
                    evidence.append(evidence_items)
                
                log_hazards = np.array(log_hazards)
                
                # Compute survival metrics
                time_points = np.array([365, 730, 1095])  # 1, 2, 3 years in days
                survival_probs = self.compute_survival_curves(log_hazards, time_points)
                risks = 1.0 - survival_probs  # Risk = 1 - Survival
                median_survival_times = self.compute_median_survival(log_hazards)
                
                logger.info(f"✅ Survival inference complete. Median survival times={median_survival_times}, Evidence extracted")
                
                return Metadata.create_agent_output(
                    status="success",
                    dataset={
                        "risk_scores": {
                            "saved_path": None,
                            "description": "Risk scores for each patient at 1, 2, and 3 years",
                            "configuration": {
                                "value": risks.tolist(),
                                "num_samples": len(risks),
                                "years": [1, 2, 3]
                            }
                        },
                        "log_hazards": {
                            "saved_path": None,
                            "description": "Log-hazard predictions from Cox model",
                            "configuration": {
                                "value": log_hazards.tolist(),
                                "num_samples": len(log_hazards)
                            }
                        },
                        "median_survival_time": {
                            "saved_path": None,
                            "description": "Median survival time for each patient (in years)",
                            "configuration": {
                                "value": median_survival_times.tolist(),
                                "num_samples": len(median_survival_times)
                            }
                        },
                        "survival_curves": {
                            "saved_path": None,
                            "description": "Survival probabilities at 1, 2, and 3 years",
                            "configuration": {
                                "value": survival_probs.tolist(),
                                "num_samples": len(survival_probs),
                                "time_points_years": [1, 2, 3]
                            }
                        },
                        "evidence": {
                            "saved_path": None,
                            "description": "Evidence sentences for each prediction based on attention weights",
                            "configuration": {
                                "value": evidence,
                                "num_samples": len(evidence)
                            }
                        }
                    },
                    model={},
                    cache_directory=self.output_dir
                )
            
            else:
                logger.info("Running classification inference")
                predictions, risk_scores, ground_truths, all_weights, evidence = [], [], [], [], []

                for emb, label, sent in zip(test_embeddings, test_labels, test_sentences):
                    emb_array = np.array(emb)[:256,:]
                    x = torch.tensor(emb_array, dtype=torch.float32).unsqueeze(0).to(device)
                    x_full = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _, _ = model(x)
                        _, attn_weights, _ = model(x_full)
            
                    prob = torch.sigmoid(logits).item()
                    pred = 1 if prob > 0.5 else 0
                    predictions.append(pred)
                    risk_scores.append(prob)
                    ground_truths.append(label)
                    all_weights.append(attn_weights.squeeze(0).cpu().tolist())

                    # Baseline risk score
                    baseline_risk = prob
                    
                    # Step 1: Identify high-attention sentences (attention > mean)
                    attn_np = attn_weights.squeeze(0).cpu().numpy()
                    mean_val = attn_np.mean()
                    high_attn_idx = np.where(attn_np > mean_val)[0]
                    
                    # Step 2: Perform ablation test on each high-attention sentence
                    candidate_evidence = []
                    for idx in high_attn_idx:
                        if idx < len(sent):
                            # Perform ablation test: remove this sentence and re-run prediction
                            emb_ablated = [emb[i] for i in range(len(emb)) if i != idx]
                            if len(emb_ablated) > 0:
                                emb_ablated_array = np.array(emb_ablated)
                                x_ablated = torch.tensor(emb_ablated_array, dtype=torch.float32).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    logits_ablated, _, _ = model(x_ablated)
                                prob_ablated = torch.sigmoid(logits_ablated).item()
                                delta_risk = baseline_risk - prob_ablated
                            else:
                                delta_risk = 0.0
                            
                            if isinstance(sent[idx], dict):
                                # New structured format with metadata
                                candidate_evidence.append({
                                    'text': sent[idx]['text'],
                                    'note_idx': sent[idx]['note_idx'],
                                    'sentence_idx': sent[idx]['sentence_idx'],
                                    'source_paragraph': sent[idx].get('source_paragraph', 0),
                                    'attention_weight': float(attn_weights.squeeze(0).cpu().numpy()[idx]),
                                    'delta_risk_score': float(delta_risk)
                                })
                            else:
                                # Backward compatibility: plain string
                                candidate_evidence.append({
                                    'text': sent[idx],
                                    'attention_weight': float(attn_weights.squeeze(0).cpu().numpy()[idx]),
                                    'delta_risk_score': float(delta_risk)
                                })
                    
                    # Step 2.5: Deduplicate similar sentences
                    candidate_evidence = self.deduplicate_evidence(candidate_evidence)
                    
                    # Step 3: Sort by delta_risk and select top k
                    candidate_evidence.sort(key=lambda x: abs(x['delta_risk_score']), reverse=True)
                    evidence_items = candidate_evidence[:top_k]
                    evidence.append(evidence_items)
                logger.info(f"✅ Inference complete: Predictions={predictions[:5]}, Risk scores={risk_scores[:5]}, Evidence={evidence[:1]}")
                return Metadata.create_agent_output(
                    status="success",
                    dataset={
                        "prediction": {
                            "saved_path": None,
                            "description": "Prediction results (class labels)",
                            "configuration": {
                                "value": predictions,
                                "num_samples": len(predictions)
                            }
                        },
                        "risk_score": {
                            "saved_path": None,
                            "description": "Risk scores for each prediction (probability of positive class)",
                            "configuration": {
                                "value": risk_scores,
                                "num_samples": len(risk_scores)
                            }
                        },
                        "evidence": {
                            "saved_path": None,
                            "description": "Evidence sentences for each prediction based on attention weights",
                            "configuration": {
                                "value": evidence,
                                "num_samples": len(evidence)
                            }
                        }
                    },
                    model={},
                    cache_directory=self.output_dir
                )

        except Exception as e:
            import traceback
            logger.error(f"❌ Inference failed: {e}")
            logger.error(traceback.format_exc())
            return Metadata.create_agent_output(
                status="error",
                dataset={
                    "error": {
                        "saved_path": None,
                        "description": str(e),
                        "configuration": {}
                    }
                },
                model={},
                cache_directory=self.output_dir
            )


if __name__ == "__main__":
    # Test the note inference tool
    from cerebra.agents.data_agent import DataAgent
    year = 1
    time_to_event = False
    dedup = True
    top_k = 15
    data_agent = DataAgent()
    input_metadata = data_agent.run(f"Load initial data for note_agent", agent_name='note_agent', patient_id=66, year=year, time_to_event=time_to_event)
    test_features_path = input_metadata.get_metadata_info()['dataset']['test_features']['saved_path']
    test_labels_path = input_metadata.get_metadata_info()['dataset']['test_labels']['saved_path']
    test_sentences_path = input_metadata.get_metadata_info()['dataset']['test_sentences']['saved_path']
    trained_model_path = input_metadata.get_metadata_info()['model']['trained_model']['saved_path']
    embedding_dim = input_metadata.get_metadata_info()['model']['trained_model']['configuration']['training_hyperparameters']['embedding_dim']
    task_type = "survival" if time_to_event else "classification"
    save_name = "note_trained_model_survival" if time_to_event else f"note_trained_model_classification"
    tool = Note_Model_Inference_Tool()  
    tool.execute(
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        test_sentences_path=test_sentences_path,
        trained_model_path=trained_model_path,
        embedding_dim=embedding_dim,
        task_type=task_type,
        top_k=top_k,
        save_name=save_name
    )