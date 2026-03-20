import os
import pickle
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from cerebra.tools.base import BaseTool
from cerebra.utils.metadata import Metadata
from cerebra.utils.log_utils import setup_logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")

# Setup logger
log_dir = os.path.join(CACHE_DIR, "image_agent", "logs")
logger = setup_logger(log_dir)


class Image_Model_Inference_Tool(BaseTool):
    def __init__(self):
        super().__init__()
        self.output_dir = os.path.join(CACHE_DIR, "image_agent", "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)

        self.set_metadata(
            tool_name="Image_Model_Inference_Tool",
            tool_description="Run inference using trained XGBoost model on brain volume features with predictions, risk scores, and top important brain regions.",
            tool_version="2.0.0",
            input_types={
                "test_data_path": "str - Path to test volume features (.csv, .pkl)",
                "trained_model_path": "str - Path to trained XGBoost model (.pkl)",
                "top_k": "int - Number of top important brain regions to return (default: 10)"
            },
            demo_commands=[
                {
                    "command": (
                        "result = tool.execute(\n"
                        "    test_data_path=dataset['test_data']['saved_path'],\n"
                        "    trained_model_path=model['xgb_model']['saved_path'],\n"
                        "    top_k=10\n"
                        ")"
                    ),
                    "description": "Run inference on brain volume test data with predictions, risk scores, and visualization."
                }
            ],
            output_type="Metadata - Inference results including predictions, risk scores, top-k important regions, and MRI visualizations",
            evaluation_criteria={
                "technical_success": "Does the tool execute without errors?",
                "inference_success": "Are predictions and risk scores provided?",
                "evidence_quality": "Are top important brain regions extracted?",
                "visualization": "Is the MRI visualization generated?"
            },
        )

    def load_data(self, data_path):
        """Load dataset from CSV or pickle file."""
        if data_path.endswith('.csv'):
            try:
                return pd.read_csv(data_path, sep='\t')
            except:
                return pd.read_csv(data_path)
        elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
            with open(data_path, 'rb') as f:
                df = pickle.load(f)
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("Pickle file should contain a DataFrame")
                return df
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def _get_average_feature_volume(self):
        region_average_volume = {
            "3rd-Ventricle": 0.001045,
            "4th-Ventricle": 0.001204,
            "Brain-Stem": 0.016520,
            "CSF": 0.190647,
            "Left-Accumbens-area": 0.000393,
            "Left-Amygdala": 0.001323,
            "Left-Caudate": 0.002304,
            "Left-Cerebellum-Cortex": 0.040404,
            "Left-Cerebellum-White-Matter": 0.010831,
            "Left-Cerebral-Cortex": 0.170791,
            "Left-Cerebral-White-Matter": 0.145676,
            "Left-Hippocampus": 0.002853,
            "Left-Inf-Lat-Vent": 0.000362,
            "Left-Lateral-Ventricle": 0.012158,
            "Left-Pallidum": 0.000775,
            "Left-Putamen": 0.005237,
            "Left-Thalamus": 0.004708,
            "Left-VentralDC": 0.002829,
            "Right-Accumbens-area": 0.000484,
            "Right-Amygdala": 0.001204,
            "Right-Caudate": 0.002512,
            "Right-Cerebellum-Cortex": 0.052035,
            "Right-Cerebellum-White-Matter": 0.010451,
            "Right-Cerebral-Cortex": 0.183210,
            "Right-Cerebral-White-Matter": 0.145305,
            "Right-Hippocampus": 0.002516,
            "Right-Inf-Lat-Vent": 0.000364,
            "Right-Lateral-Ventricle": 0.011487,
            "Right-Pallidum": 0.001019,
            "Right-Putamen": 0.003402,
            "Right-Thalamus": 0.004831,
            "Right-VentralDC": 0.002709,
        }
        return region_average_volume

    def _prepare_features(self, test_df, feature_cols):
        """Prepare and normalize features from test data - matches training preprocessing."""
        available_features = [col for col in feature_cols if col in test_df.columns]
        X_test_df = test_df[available_features].copy()
        
        # First pass: handle string/object types and infinite values
        for col in available_features:
            if col in X_test_df.columns:
                # Convert string values to numeric if needed
                if X_test_df[col].dtype == 'object':
                    X_test_df[col] = X_test_df[col].apply(
                        lambda x: float(str(x).strip('[]')) if isinstance(x, str) else x
                    )
                
                # Convert to numeric
                X_test_df[col] = pd.to_numeric(X_test_df[col], errors='coerce')
                
                # Replace infinite values with NaN
                X_test_df[col] = X_test_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Handle ICV and normalize BEFORE imputation and clipping
        if 'ICV' in test_df.columns:
            icv_values = test_df['ICV'].copy()
            
            # Replace infinite values with NaN
            icv_values = icv_values.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with median
            if icv_values.isnull().any():
                icv_median = icv_values.median()
                if pd.isna(icv_median):
                    icv_median = 1.0  # Fallback to avoid division by zero
                icv_values = icv_values.fillna(icv_median)
            
            # Clip ICV to avoid division by zero
            icv_values = icv_values.clip(lower=1e-6)
            
            # Normalize features by ICV FIRST
            X_test_df = X_test_df.div(icv_values.values, axis=0)
            
            # Handle any resulting infinite values after normalization
            for col in available_features:
                X_test_df[col] = X_test_df[col].replace([np.inf, -np.inf], np.nan)
        
        # NOW impute and clip on normalized data
        for col in available_features:
            if col in X_test_df.columns:
                # Fill NaN with median of NORMALIZED values
                if X_test_df[col].isnull().any():
                    median_val = X_test_df[col].median()
                    # If all values are NaN, use 0
                    if pd.isna(median_val):
                        median_val = 0
                    X_test_df[col] = X_test_df[col].fillna(median_val)
                
                # Clip extreme values on NORMALIZED scale
                upper_bound = X_test_df[col].quantile(0.999)
                lower_bound = X_test_df[col].quantile(0.001)
                X_test_df[col] = X_test_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Convert to numpy array and ensure float32 dtype
        X_test = X_test_df.fillna(0).values.astype(np.float32)
        
        return X_test, available_features

    def _compute_feature_importance(self, model, X_test, available_features, top_k):
        """
        Compute per-sample feature importance using ablation (prediction change) method.
        Measures how much each feature affects the prediction when removed.
        """
        try:
            import xgboost as xgb
            
            logger.info("Computing feature importance via ablation (prediction change)...")
            per_sample_importance = []
            
            for i in range(X_test.shape[0]):
                sample = X_test[i:i+1]
                
                # Get baseline prediction (risk score)
                if hasattr(model, 'predict_proba'):
                    baseline_risk = model.predict_proba(sample)[:, 1][0]
                else:
                    dtest = xgb.DMatrix(sample)
                    baseline_risk = -model.predict(dtest)[0]
                
                # Compute importance by ablating each feature
                feature_importances = []
                for idx in range(len(available_features)):
                    # Create ablated version with this feature set to 0
                    sample_ablated = sample.copy()
                    sample_ablated[0, idx] = 0
                    
                    # Get prediction without this feature
                    if hasattr(model, 'predict_proba'):
                        ablated_risk = model.predict_proba(sample_ablated)[:, 1][0]
                    else:
                        dtest_ablated = xgb.DMatrix(sample_ablated)
                        ablated_risk = -model.predict(dtest_ablated)[0]
                    
                    # Importance is the change in prediction (delta risk)
                    delta_risk = baseline_risk - ablated_risk
                    
                    feature_importances.append({
                        'feature': available_features[idx],
                        'feature_importance': float(delta_risk),
                        'feature_value': float(sample[0, idx]),
                        'average_feature_volume': float(self._get_average_feature_volume()[available_features[idx]])
                    })
                
                # Sort by absolute delta_risk and take top k
                feature_importances.sort(key=lambda x: abs(x['feature_importance']), reverse=True)
                sample_top_features = feature_importances[:top_k]
                
                per_sample_importance.append(sample_top_features)
            
            logger.info(f"✅ Computed ablation-based feature importance for {len(per_sample_importance)} samples")
            return per_sample_importance
            
        except Exception as e:
            logger.warning(f"⚠️ Feature importance computation failed: {e}, returning empty importance")
            import traceback
            logger.debug(traceback.format_exc())
            return [[] for _ in range(X_test.shape[0])]



    def visualize_mri(self, subject_id, session_id, output_path, top_regions=None):
        """
        Visualize MRI scan only (no segmentation overlays).

        Args:
            subject_id: Patient/subject ID
            session_id: Session ID
            output_path: Path to save the visualization
            top_regions: Unused, kept for backward compatibility

        Returns:
            Path to saved visualization or empty string if failed
        """
        try:
            mri_base_path = os.environ.get("MRI_BASE_PATH")
            if not mri_base_path:
                logger.warning("⚠️ MRI_BASE_PATH is not set in environment")
                return ''

            mri_filename = f"{subject_id}_{session_id}_mri.mgz"
            mri_path = os.path.join(mri_base_path, mri_filename)

            if not os.path.exists(mri_path):
                logger.warning(f"⚠️ MRI file not found: {mri_path}")
                return ''

            mri_img = nib.load(mri_path).get_fdata()

            # Create figure with 3x3 subplots for sagittal/coronal/axial views.
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))

            center_x = mri_img.shape[0] // 2
            center_y = mri_img.shape[1] // 2
            center_z = mri_img.shape[2] // 2

            x_slices = [max(0, center_x - 10), center_x, min(mri_img.shape[0] - 1, center_x + 10)]
            y_slices = [max(0, center_y - 10), center_y, min(mri_img.shape[1] - 1, center_y + 10)]
            z_slices = [max(0, center_z - 10), center_z, min(mri_img.shape[2] - 1, center_z + 10)]

            for i, x_slice in enumerate(x_slices):
                ax = axes[0, i]
                ax.imshow(np.rot90(mri_img[x_slice, :, :]), cmap='gray')
                ax.set_title(f'Sagittal Slice {i+1}')
                ax.axis('off')

            for i, y_slice in enumerate(y_slices):
                ax = axes[1, i]
                ax.imshow(np.rot90(mri_img[:, y_slice, :]), cmap='gray')
                ax.set_title(f'Coronal Slice {i+1}')
                ax.axis('off')

            for i, z_slice in enumerate(z_slices):
                ax = axes[2, i]
                ax.imshow(np.rot90(mri_img[:, :, z_slice]), cmap='gray')
                ax.set_title(f'Axial Slice {i+1}')
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"✅ MRI visualization saved to {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"⚠️ MRI visualization failed for {subject_id}_{session_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ''


    def execute(self,
                test_data_path: str,
                trained_model_path: str,
                top_k: int = 10,
                cache_dir: str = None):
        try:
            # Load data and model
            test_df = self.load_data(test_data_path)
            with open(trained_model_path, 'rb') as f:
                model_dict = pickle.load(f)
                model, feature_cols = model_dict['model'], model_dict['feature_cols']
            
            # Prepare features
            X_test, available_features = self._prepare_features(test_df, feature_cols)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                predictions = model.predict(X_test)
                risk_scores = model.predict_proba(X_test)[:, 1]
            else:
                import xgboost as xgb
                dtest = xgb.DMatrix(X_test)
                predictions = model.predict(dtest)
                risk_scores = -predictions
            
            # Compute feature importance
            per_sample_importance = self._compute_feature_importance(model, X_test, available_features, top_k)
            
            # Generate MRI visualizations
            cache_dir = cache_dir or os.path.join(CACHE_DIR, "image_agent", "evaluation")
            os.makedirs(cache_dir, exist_ok=True)
            unique_id = str(uuid.uuid4())[:8]
            
            viz_paths = []
            for idx, row in test_df.iterrows():
                if row.get('subject_id') and row.get('session_id'):
                    viz_path = os.path.join(cache_dir, f"mri_viz_{unique_id}_{idx}.png")
                    # Pass top important regions for this sample to the visualization
                    sample_idx = list(test_df.index).index(idx)
                    top_regions = per_sample_importance[sample_idx] if sample_idx < len(per_sample_importance) else None
                    viz_paths.append(self.visualize_mri(row['subject_id'], row['session_id'], viz_path, top_regions))
                else:
                    viz_paths.append('')
            logger.info(f"✅ Inference complete: Predictions={predictions}, Risk scores={risk_scores}, Evidence={per_sample_importance}, Visualiation Paths={viz_paths}")
            
            # Return results
            return Metadata.create_agent_output(
                status="success",
                dataset={
                    "prediction": {
                        "saved_path": None,
                        "description": "Prediction results",
                        "configuration": {"value": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions), "num_samples": len(predictions)}
                    },
                    "risk_score": {
                        "saved_path": None,
                        "description": "Risk scores",
                        "configuration": {"value": risk_scores.tolist() if hasattr(risk_scores, 'tolist') else list(risk_scores), "num_samples": len(risk_scores)}
                    },
                    "evidence": {
                        "saved_path": None,
                        "description": f"Top {top_k} important brain regions per sample",
                        "configuration": {"value": per_sample_importance, "num_samples": len(per_sample_importance)}
                    },
                    "mri_visualizations": {
                        "saved_path": None,
                        "description": "MRI visualizations with segmentation",
                        "configuration": {"value": viz_paths, "num_visualizations": len([p for p in viz_paths if p])}
                    },
                },
                model={},
                cache_directory=self.output_dir
            )

        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return Metadata.create_agent_output(
                status="error",
                dataset={"error": {"saved_path": None, "description": str(e), "configuration": {}}},
                model={},
                cache_directory=self.output_dir
            )

if __name__ == "__main__":
    # Test the XGBoost inference tool
    tool = Image_Model_Inference_Tool()
    
    # Example paths (update with your actual paths)
    test_data_path = 'path/to/X_mri_volume_test.csv'
    trained_model_path = 'path/to/trained_image_model_classification_prediction_1yr.pkl'
    
    result = tool.execute(
        test_data_path=test_data_path,
        trained_model_path=trained_model_path,
        top_k=10
    )

    metadata_dict = result.get_metadata_info()
    if metadata_dict["status"] == "success":
        print("✅ Image inference completed successfully!")
        print(f"Number of predictions: {metadata_dict['dataset']['prediction']['configuration']['num_samples']}")
        
        # Show top important regions for first sample
        if len(metadata_dict['dataset']['top_important_regions']['configuration']['value']) > 0:
            print(f"\nTop important regions for first sample:")
            for region in metadata_dict['dataset']['top_important_regions']['configuration']['value'][0][:5]:
                print(f"  - {region['feature']}: importance={region['importance']:.4f}, value={region['feature_value']:.4f}")
        
        print(f"\nNumber of visualizations: {metadata_dict['dataset']['mri_visualizations']['configuration']['num_visualizations']}")
    else:
        print(f"❌ Image inference failed: {metadata_dict['dataset']['error']['description']}")