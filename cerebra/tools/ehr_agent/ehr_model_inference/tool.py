import os
import pickle
import joblib
import numpy as np
from tqdm import tqdm
from scipy.sparse import vstack
from cerebra.tools.base import BaseTool
from cerebra.utils.metadata import Metadata
from cerebra.utils.log_utils import setup_logger
from cerebra.utils.ehr_headers import load_ehr_headers
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")
# Setup logger
log_dir = os.path.join(CACHE_DIR, "ehr_agent", "logs")
logger = setup_logger(log_dir)

class EHR_Model_Inference_Tool(BaseTool):
    def __init__(self):
        super().__init__()
        self.output_dir = os.path.join("cerebra_cache", "ehr_agent", "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)

        self.set_metadata(
            tool_name="EHR_Model_Inference_Tool",
            tool_description="Evaluate a trained ML model (classification or survival) on EHR test data and return predictions with confidence scores/survival times and metrics.",
            tool_version="1.0.0",
            input_types={
                "test_data_path": "str - Path to test features (.npz, .pkl)",
                "trained_model_path": "str - Path to trained model (.joblib)",
            },
            demo_commands=[
                {
                    "command": (
                        "result = tool.execute(\n"
                        "    test_data_path=dataset['test_data']['saved_path'],\n"
                        "    trained_model_path=model['trained_model']['saved_path'],\n"
                        "    ehr_header_path=dataset.get('ehr_headers', {}).get('saved_path')\n"
                        ")"
                    ),
                    "description": "Run inference on EHR test set with predictions and risk scores/median survival times."
                }
            ],
            output_type="Metadata - Inference results including metrics, predictions, and risk scores/survival times",
            evaluation_criteria={
                "technical_success": "Does the tool execute without errors?",
                "inference_success": "Is the prediction provided?",
                "evidence_quality": "Are the evidence sentences extracted?",
                "risk_score": "Is the risk score or median survival time provided?"
            }
        )
    
    def is_survival_model(self, model):
        """Check if the model is a survival model (RandomSurvivalForest)"""
        try:
            from sksurv.ensemble import RandomSurvivalForest
            return isinstance(model, RandomSurvivalForest)
        except ImportError:
            return False
    
    def get_evidence_classification(self, model, X_test, top_k=10, ehr_header_path=None):
        """
        Get evidence for classification models using gradient-based feature importance.
        Alternative to SHAP when it returns zeros.
        """
        ehr_headers = load_ehr_headers(
            n_features=X_test.shape[1],
            explicit_path=ehr_header_path,
        )

        if hasattr(X_test, 'toarray'):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = X_test
                
        # Get predictions
        predicted_class = model.predict(X_test)
        risk_scores = model.predict_proba(X_test)[:, 1]
        
        # Compute feature importance using gradient-based method
        logger.info("Computing gradient-based feature importance for classification...")
        try:
            # Get global feature importance from the model
            if hasattr(model, 'feature_importances_'):
                global_importance = model.feature_importances_
                logger.info("Using model's built-in feature_importances_")
            elif hasattr(model, 'coef_'):
                # For linear models
                global_importance = np.abs(model.coef_[0])
                logger.info("Using model's coefficients for feature importance")
            else:
                # Fallback: use uniform importance
                global_importance = np.ones(X_test_dense.shape[1])
                logger.info("Using uniform feature importance (fallback)")
            
            # Normalize global importance
            if global_importance.sum() > 0:
                global_importance = global_importance / global_importance.sum()
            else:
                global_importance = np.ones_like(global_importance) / len(global_importance)
            
            # Compute per-sample importance
            evidences_all = []
            for i in range(len(predicted_class)):
                sample = X_test_dense[i]
                
                # Weight global importance by absolute feature values
                sample_importance = np.abs(sample) * global_importance
                
                # Normalize to sum to 1 (only for non-zero features)
                if sample_importance.sum() > 0:
                    sample_importance = sample_importance / sample_importance.sum()
                
                # Get top-k features (only non-zero ones)
                nonzero_mask = sample != 0
                sample_importance_nonzero = sample_importance.copy()
                sample_importance_nonzero[~nonzero_mask] = 0
                
                top_indices = np.argsort(sample_importance_nonzero)[::-1][:top_k]
                
                # Filter to only include non-zero features
                top_indices = [idx for idx in top_indices if sample[idx] != 0]
                
                # Create evidence strings
                evidences = []
                for j in top_indices:
                    if j < len(ehr_headers):  # Safety check
                        evidences.append(
                            f"Feature in EHR: {ehr_headers[j]}, corresponding value: {X_test_dense[i, j]}. Feature importance: {sample_importance_nonzero[j]}. "
                        )
                evidences_all.append(evidences)
            
            logger.info(f"✅ Feature importance computed for {len(evidences_all)} samples")
            
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}. Using fallback method.")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Fallback: just use non-zero features ordered by value
            evidences_all = []
            for i in range(len(predicted_class)):
                nonzero_idx = np.nonzero(X_test_dense[i])[0]
                # Sort by absolute value
                if len(nonzero_idx) > 0:
                    sorted_indices = nonzero_idx[np.argsort(np.abs(X_test_dense[i, nonzero_idx]))[::-1]][:top_k]
                else:
                    sorted_indices = []
                
                evidences = []
                for j in sorted_indices:
                    if j < len(ehr_headers):  # Safety check
                        evidences.append(
                            f"Feature in EHR: {ehr_headers[j]}, corresponding value: {X_test_dense[i, j]}. "
                        )
                evidences_all.append(evidences)
        
        return predicted_class, risk_scores, evidences_all

    def get_evidence_survival(self, model, X_test, top_k=10, ehr_header_path=None):
        """Get evidence for survival models using feature perturbation for single example and predict median survival time"""
        
        ehr_headers = load_ehr_headers(
            n_features=X_test.shape[1],
            explicit_path=ehr_header_path,
        )

        # Convert to CSR for efficient slicing if sparse
        if hasattr(X_test, 'toarray'):
            from scipy.sparse import csr_matrix
            X_test_csr = csr_matrix(X_test)
        else:
            X_test_csr = X_test
        
        # Get survival functions for each sample
        survival_functions = model.predict_survival_function(X_test_csr)
        
        # Extract median survival time from each survival function
        median_survival_times = []
        for surv_func in survival_functions:
            # Get the time when survival probability crosses 0.5
            times = surv_func.x
            probabilities = surv_func.y
            
            # Find the time where survival probability drops below 0.5
            median_time = None
            for t, prob in zip(times, probabilities):
                if prob <= 0.5:
                    median_time = t
                    break
            
            # If survival never drops to 0.5, use the last time point
            if median_time is None:
                median_time = times[-1] if len(times) > 0 else np.inf
            
            median_survival_times.append(median_time)
        
        median_survival_times = np.array(median_survival_times)
        
        # Convert to dense for feature extraction
        if hasattr(X_test_csr, 'toarray'):
            X_test_dense = X_test_csr.toarray()
        else:
            X_test_dense = X_test_csr
        
        # Compute local feature importance via perturbation for single example
        logger.info("Computing local feature importance via perturbation...")
        
        evidences_all = []
        for i in range(len(X_test_dense)):
            # Get baseline prediction (risk score)
            baseline_prediction = model.predict(X_test_csr[i:i+1])[0]
            
            # Get all features for this sample
            # Get nonzero features for this sample
            nonzero_idx = np.nonzero(X_test_dense[i])[0]
            
            # Compute importance by perturbing each nonzero feature
            feature_importances = []
            for idx in nonzero_idx:
                # Create perturbed version with this feature zeroed out
                if hasattr(X_test_csr, 'toarray'):
                    from scipy.sparse import lil_matrix
                    X_perturbed = lil_matrix(X_test_csr[i:i+1].copy())
                    X_perturbed[0, idx] = 0
                    X_perturbed = X_perturbed.tocsr()
                else:
                    X_perturbed = X_test_dense[i:i+1].copy()
                    X_perturbed[0, idx] = 0
                
                # Get prediction without this feature
                perturbed_prediction = model.predict(X_perturbed)[0]
                
                # Importance is the absolute change in prediction
                importance = abs(perturbed_prediction - baseline_prediction)
                if importance > 0:
                    feature_importances.append((idx, perturbed_prediction - baseline_prediction))
            
            # Sort by importance and take top_k
            feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)
            top_indices = feature_importances[:top_k]

            logger.info(f"Local feature importance computed for sample {i}")
            # Create evidence strings
            evidences = []
            for j, importance in top_indices:
                evidences.append(
                    f"Feature in EHR: {ehr_headers[j]}, corresponding value: {X_test_dense[i, j]}. "
                    # f"Feature Importance Score: {importance:.4f}. "
                )
            evidences_all.append(evidences)
        # For survival models, return median survival times, survival functions, and evidence
        return None, median_survival_times, evidences_all, survival_functions

    def execute(self,
                test_data_path: str,
                trained_model_path: str,
                top_k: int = 15,
                ehr_header_path: str = None,
                **kwargs):
        try:
        # Load inputs
            def load_data(path):
                if path.endswith(".npz"):
                    from scipy.sparse import load_npz
                    return load_npz(path)
                elif path.endswith(".npy"):
                    return np.load(path)
                elif path.endswith(".pkl"):
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {path}")

            X_test = load_data(test_data_path)
            model = joblib.load(trained_model_path)

            logger.info(f"Loaded model from {trained_model_path}")
            logger.info(f"Test data shape: {len(X_test)}")

            temporal_processed_data = []
            for data in X_test:
                temporal_processed_data.append(data.max(axis=0))

            X_test_processed = vstack(temporal_processed_data)
            
            # Check if survival or classification model
            is_survival = self.is_survival_model(model)
            
            if is_survival:
                logger.info("Detected survival model, using survival inference method")
                predictions, median_survival_times, evidence, survival_functions = self.get_evidence_survival(
                    model, X_test_processed, top_k=top_k, ehr_header_path=ehr_header_path
                )
                years = np.array([365, 730, 1095], dtype=float)
                risks = np.vstack([1.0 - sf(years) for sf in survival_functions])
                median_survival_times = median_survival_times/365.0 # convert to years
                logger.info(f"✅ Inference complete: Median survival times={median_survival_times}, Evidence={evidence}")
            else:
                logger.info("Detected classification model, using classification inference method")
                predictions, risk_scores, evidence = self.get_evidence_classification(
                    model, X_test_processed, top_k=top_k, ehr_header_path=ehr_header_path
                )
                logger.info(f"✅ Inference complete: Predictions={predictions}, Risk scores={risk_scores}, Evidence={evidence}")

            # Save results
            if is_survival:
                dataset = {
                    "risk_scores": {
                        "saved_path": None,
                        "description": "Risk scores for each patient at 1, 2, and 3 years",
                        "configuration": {
                            "value": risks.tolist(),
                            "num_samples": len(risks),
                            "years": (years/365.0).tolist()
                        }
                    },
                    "survival_functions": {
                        "saved_path": None,
                        "description": "Survival functions for each patient",
                        "configuration": {
                            "value": [(sf.x.tolist(), (1-sf.y).tolist()) for sf in survival_functions],
                            "num_samples": len([(sf.x.tolist(), (1-sf.y).tolist()) for sf in survival_functions])
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
                    "evidence": {
                        "saved_path": None,
                        "description": "Evidence sentences for each prediction",
                        "configuration": {
                            "value": evidence,
                            "num_samples": len(evidence)
                        }
                    }
                }
            else:
                dataset = {
                    "prediction": {
                        "saved_path": None,
                        "description": "Prediction results (class labels)",
                        "configuration": {
                            "value": predictions.tolist(),
                            "num_samples": len(predictions),
                            "model_type": "classification"
                        }
                    },
                    "risk_score": {
                        "saved_path": None,
                        "description": "Risk scores for each prediction (probability of positive class)",
                        "configuration": {
                            "value": risk_scores.tolist(),
                            "num_samples": len(risk_scores)
                        }
                    },
                    "evidence": {
                        "saved_path": None,
                        "description": "Evidence sentences for each prediction",
                        "configuration": {
                            "value": evidence,
                            "num_samples": len(evidence)
                        }
                    }
                }

            return Metadata.create_agent_output(
                status="success",
                dataset=dataset,
                model={},
                cache_directory=self.output_dir
            )

        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            return Metadata.create_agent_output(
                status="error",
                model={
                    "trained_model": {"saved_path": trained_model_path, "description": "", "configuration": {}}
                },
                dataset={
                    "error": {"saved_path": None, "description": str(e), "configuration": {}}
                },
                cache_directory=self.output_dir
            )

if __name__ == "__main__":
    from cerebra.agents.data_agent import DataAgent
    from tqdm import tqdm
    year = 1

    curves_all = []

    for patient_id in tqdm(range(1, 200)):
        data_agent = DataAgent()
        time_to_event = True
        input_metadata = data_agent.run(f"Load initial data for ehr_agent", agent_name='ehr_agent', patient_id=patient_id, year=year, time_to_event=time_to_event)
        test_data_path = input_metadata.get_metadata_info()['dataset']['test_data']['saved_path']
        trained_model_path = 'path/to/trained_model.joblib'
        tool = EHR_Model_Inference_Tool()
        result = tool.execute(test_data_path=test_data_path, trained_model_path=trained_model_path)
        curves = result.get_metadata_info()['dataset']['survival_functions']['configuration']['value']
        curves_all.append(curves)
    
    import matplotlib.pyplot as plt

    plt.figure()
    for curves in curves_all:
        for curve in curves:
            plt.plot(curve[0], curve[1])
    plt.show()
    plt.savefig('survival_curves.png')