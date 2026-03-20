import os
import pickle
import numpy as np
import pandas as pd
from cerebra.utils.metadata import Metadata
from tqdm import tqdm
from typing import List, Optional, Dict, Union
from cerebra.tools.base import BaseTool
from cerebra.utils.log_utils import setup_logger
import traceback
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
import xgboost as xgb
from lifelines.utils import concordance_index
from itertools import product
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")
# Configure logging to save to the tool's cache directory
log_dir = os.path.join(CACHE_DIR, "image_agent", "logs")
logger = setup_logger(log_dir)


class Image_Model_Trainer_Tool(BaseTool):
    """
    An end-to-end tool that trains XGBoost models on brain volume features
    extracted from MRI scans for classification.
    """

    def __init__(self):
        super().__init__()

        # Output directories
        self.models_output_dir = os.path.join(self.output_dir or CACHE_DIR, "image_agent", "models")
        os.makedirs(self.models_output_dir, exist_ok=True)

        self.set_metadata(
            tool_name="Image_Model_Trainer_Tool",
            tool_description="End-to-end tool that trains XGBoost models on brain volume features for classification",
            tool_version="3.0.0",
            input_types={
                "train_data_path(required)": "str - Path to CSV/pickle file containing training volume features",
                "train_labels_path(required)": "str - Path to numpy file containing training labels",
                "validation_data_path(required)": "str - Path to CSV/pickle file containing validation volume features",
                "validation_labels_path(required)": "str - Path to numpy file containing validation labels",
                "algorithm": "str - Algorithm to use for training ('xgboost', 'xgboost_survival')",
                "save_name": "str - Name for saving the trained model (default: 'xgb_volume_model')"
            },
            output_type="Metadata - Metadata object containing trained model path, validation AUROC, and training metrics",
            demo_commands=[
                {
                    "command": (
                        "result = tool.execute(\n"
                        "    train_data_path=dataset['train_data']['saved_path'],\n"
                        "    train_labels_path=dataset['train_labels']['saved_path'],\n"
                        "    validation_data_path=dataset['validation_data']['saved_path'],\n"
                        "    validation_labels_path=dataset['validation_labels']['saved_path'],\n"
                        "    algorithm='xgboost',\n"
                        "    save_name='xgb_volume_1yr_model'\n"
                        ")"
                    ),
                    "description": "Train XGBoost model using extracted brain volume features for classification"
                },
                {
                    "command": (
                        "result = tool.execute(\n"
                        "    train_data_path=dataset['train_data']['saved_path'],\n"
                        "    train_labels_path=dataset['train_labels']['saved_path'],\n"
                        "    validation_data_path=dataset['validation_data']['saved_path'],\n"
                        "    validation_labels_path=dataset['validation_labels']['saved_path'],\n"
                        "    algorithm='xgboost_survival',\n"
                        "    save_name='xgb_volume_survival_model'\n"
                        ")"
                    ),
                    "description": "Train XGBoost model using extracted brain volume features for survival"
                }
            ],
            user_metadata={
                "limitations": [
                    "Requires brain volume features extracted from MRI (e.g., from synthseg)",
                    "Limited to binary classification tasks",
                    "Expects data with ICV column for normalization"
                ],
                "best_practices": [
                    "Normalize volume features by ICV for better performance",
                    "Use grid search for hyperparameter tuning on new datasets",
                    "Monitor training/validation curves for overfitting",
                    "Validate feature columns exist before training"
                ]
            },
            evaluation_criteria={
                "technical_success": "Did the tool execute without errors?",
                "performance_evaluation": "Is the model AUROC satisfactory (>0.65)?",
                "improvement_opportunities": "What specific hyperparameters should be adjusted?",
                "efficiency": "Training should complete quickly given tabular data nature"
            }
        )

    def load_data(self, data_path: str, labels_path: str = None):
        """Load dataset from CSV or pickle file."""
        if data_path.endswith('.csv'):
            # Try tab-separated first, then comma
            try:
                df = pd.read_csv(data_path, sep='\t')
            except:
                df = pd.read_csv(data_path)
        elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
            with open(data_path, 'rb') as f:
                df = pickle.load(f)
                if not isinstance(df, pd.DataFrame):
                    # If it's a list of paths, we can't use this format
                    raise ValueError("Pickle file should contain a DataFrame with volume features")
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Load labels if separate file provided
        if labels_path is not None:
            if labels_path.endswith('.npy'):
                labels = np.load(labels_path)
            elif labels_path.endswith('.pkl') or labels_path.endswith('.pickle'):
                with open(labels_path, 'rb') as f:
                    labels = pickle.load(f)
            else:
                labels = None
            return df, labels
        
        return df, None

    def clean_and_process_data(self, df: pd.DataFrame, normalize_by_icv: bool = True):
        """Clean and process volume data, optionally normalizing by ICV."""
        # Feature columns are all the brain volume measurements (excluding metadata)
        metadata_cols = ['fs_id', 'subject_id', 'session_id', 'split', 'label', 'ICV']
        feature_cols = [col for col in df.columns if col not in metadata_cols]

        print(f"Using {len(feature_cols)} volume features")
        df_clean = df.copy()
        
        # Impute missing values and handle infinite values
        for col in feature_cols:
            if col in df_clean.columns:
                # Replace infinite values with NaN
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN with median
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                
                # Clip extreme values
                upper_bound = df_clean[col].quantile(0.999)
                lower_bound = df_clean[col].quantile(0.001)
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

        # Handle ICV column
        if 'ICV' in df_clean.columns:
            df_clean['ICV'] = df_clean['ICV'].replace([np.inf, -np.inf], np.nan)
            if df_clean['ICV'].isnull().any():
                icv_median = df_clean['ICV'].median()
                df_clean['ICV'] = df_clean['ICV'].fillna(icv_median)
                print(f"Imputed missing ICV values with median: {icv_median}")
            
            # Normalize by ICV if requested
            if normalize_by_icv:
                df_clean['ICV'] = df_clean['ICV'].clip(lower=1e-6)
                for col in feature_cols:
                    df_clean[col] = df_clean[col] / df_clean['ICV']
                    # Handle any resulting infinite values
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], df_clean[col].median())
                print("Volume features normalized by ICV")

        print(f"Dataset size: {len(df)} -> {len(df_clean)}")
        if 'label' in df_clean.columns:
            print(f"Label distribution: {df_clean['label'].value_counts().to_dict()}")

        return df_clean, feature_cols

    def execute(self,
                train_data_path: str,
                train_labels_path: str,
                validation_data_path: str,
                validation_labels_path: str,
                algorithm: str = "xgboost",
                save_name: str = "xgb_volume_model",
                **kwargs):
        """
        Execute the complete XGBoost volume trainer: load data → preprocess → train model → evaluate.
        """
        try:
            logger.info("Initializing XGBoost Volume Model Trainer")

            # Load datasets
            logger.info("Loading training data...")
            train_df, train_labels_ext = self.load_data(train_data_path, train_labels_path)
            
            logger.info("Loading validation data...")
            val_df, val_labels_ext = self.load_data(validation_data_path, validation_labels_path)

            # Clean and process data (keep track of indices for label alignment)
            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)
            train_df_clean, feature_cols = self.clean_and_process_data(train_df, normalize_by_icv=True)
            val_df_clean, _ = self.clean_and_process_data(val_df, normalize_by_icv=True)

            # Prepare features and labels (filter labels to match cleaned data indices)
            X_train = train_df_clean[feature_cols].values
            if train_labels_ext is not None:
                train_labels_ext = np.array(train_labels_ext)  # Convert to numpy array
                train_idx = train_df_clean.index.to_numpy().astype(int)
                logger.info(f"Train labels shape: {train_labels_ext.shape}, Train indices shape: {train_idx.shape}")
                y_train = train_labels_ext[train_idx]
            else:
                y_train = train_df_clean['label'].values

            X_val = val_df_clean[feature_cols].values
            if val_labels_ext is not None:
                val_labels_ext = np.array(val_labels_ext)  # Convert to numpy array
                val_idx = val_df_clean.index.to_numpy().astype(int)
                logger.info(f"Val labels shape: {val_labels_ext.shape}, Val indices shape: {val_idx.shape}")
                y_val = val_labels_ext[val_idx]
            else:
                y_val = val_df_clean['label'].values
            

            logger.info(f"Training set size: {len(X_train)}")
            logger.info(f"Validation set size: {len(X_val)}")

            if 'survival' in algorithm.lower():
                # Survival model using XGBoost AFT
                logger.info("Training XGBoost Survival (AFT) model...")
                
                # Handle 2D labels (time, event) format
                events_train = y_train[:, 1].astype(bool) if y_train.ndim == 2 else np.ones(len(y_train), dtype=bool)
                times_train = y_train[:, 0].astype(float) if y_train.ndim == 2 else y_train.astype(float)
                events_val = y_val[:, 1].astype(bool) if y_val.ndim == 2 else np.ones(len(y_val), dtype=bool)
                times_val = y_val[:, 0].astype(float) if y_val.ndim == 2 else y_val.astype(float)
                
                # For AFT: uncensored have y_lower = y_upper = time; censored have y_lower = time, y_upper = inf
                y_lower_train = times_train.copy()
                y_upper_train = times_train.copy()
                y_upper_train[~events_train] = np.inf
                
                y_lower_val = times_val.copy()
                y_upper_val = times_val.copy()
                y_upper_val[~events_val] = np.inf
                
                # Create DMatrix with survival labels
                dtrain = xgb.DMatrix(X_train)
                dtrain.set_float_info('label_lower_bound', y_lower_train)
                dtrain.set_float_info('label_upper_bound', y_upper_train)
                
                dval = xgb.DMatrix(X_val)
                dval.set_float_info('label_lower_bound', y_lower_val)
                dval.set_float_info('label_upper_bound', y_upper_val)
                
                # Grid search for survival model
                logger.info("Performing hyperparameter grid search for survival model...")
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [2, 3],
                    'learning_rate': [0.05, 0.1]
                }
                
                best_c_index = -np.inf
                best_params = None
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                
                # Generate all parameter combinations
                param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
                
                for params in param_combinations:
                    cv_scores = []
                    for train_idx_cv, val_idx_cv in kf.split(X_train):
                        X_train_cv, X_val_cv = X_train[train_idx_cv], X_train[val_idx_cv]
                        times_train_cv, times_val_cv = times_train[train_idx_cv], times_train[val_idx_cv]
                        events_train_cv, events_val_cv = events_train[train_idx_cv], events_train[val_idx_cv]
                        
                        # Prepare CV fold data
                        y_lower_cv = times_train_cv.copy()
                        y_upper_cv = times_train_cv.copy()
                        y_upper_cv[~events_train_cv] = np.inf
                        
                        dtrain_cv = xgb.DMatrix(X_train_cv)
                        dtrain_cv.set_float_info('label_lower_bound', y_lower_cv)
                        dtrain_cv.set_float_info('label_upper_bound', y_upper_cv)
                        
                        # Train model
                        xgb_params_cv = {
                            'objective': 'survival:aft',
                            'eval_metric': 'aft-nloglik',
                            'aft_loss_distribution': 'normal',
                            'max_depth': params['max_depth'],
                            'learning_rate': params['learning_rate'],
                            'tree_method': 'hist'
                        }
                        
                        model_cv = xgb.train(
                            params=xgb_params_cv,
                            dtrain=dtrain_cv,
                            num_boost_round=params['n_estimators'],
                            verbose_eval=False
                        )
                        
                        # Evaluate on validation fold
                        pred_cv = model_cv.predict(xgb.DMatrix(X_val_cv))
                        c_index_cv = concordance_index(times_val_cv, pred_cv, events_val_cv)
                        cv_scores.append(c_index_cv)
                    
                    mean_c_index = np.mean(cv_scores)
                    if mean_c_index > best_c_index:
                        best_c_index = mean_c_index
                        best_params = params
                
                logger.info(f"Best parameters found (C-index: {best_c_index:.4f}): {best_params}")
                
                # Train final model with best params
                xgb_params = {
                    'objective': 'survival:aft',
                    'eval_metric': 'aft-nloglik',
                    'aft_loss_distribution': 'normal',
                    'max_depth': best_params['max_depth'],
                    'learning_rate': best_params['learning_rate'],
                    'tree_method': 'hist'
                }
                
                xgb_model = xgb.train(
                    params=xgb_params,
                    dtrain=dtrain,
                    num_boost_round=best_params['n_estimators'],
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=20,
                    verbose_eval=False
                )
                logger.info("XGBoost survival training complete!")
                
                # Evaluate with concordance index
                train_pred = xgb_model.predict(xgb.DMatrix(X_train))
                val_pred = xgb_model.predict(xgb.DMatrix(X_val))
                train_auc = concordance_index(times_train, train_pred, events_train)
                val_auc = concordance_index(times_val, val_pred, events_val)
                logger.info(f"Training C-index: {train_auc:.4f}, Validation C-index: {val_auc:.4f}")
                
                # Time-dependent AUC and AUPRC at 1yr, 2yr, 3yr
                time_points = [365, 730, 1095]  # 1, 2, 3 years in days
                train_time_metrics = {}
                val_time_metrics = {}
                
                for t in time_points:
                    year = int(t / 365)
                    try:
                        # Training set: binary outcome (event occurred before time t)
                        y_binary_train = (times_train <= t) & events_train
                        risk_scores_train = -train_pred  # Higher risk = shorter predicted time
                        
                        if len(np.unique(y_binary_train)) > 1:
                            train_time_metrics[f'auc_{year}yr'] = roc_auc_score(y_binary_train, risk_scores_train)
                            train_time_metrics[f'auprc_{year}yr'] = average_precision_score(y_binary_train, risk_scores_train)
                        else:
                            train_time_metrics[f'auc_{year}yr'] = np.nan
                            train_time_metrics[f'auprc_{year}yr'] = np.nan
                        
                        # Validation set
                        y_binary_val = (times_val <= t) & events_val
                        risk_scores_val = -val_pred
                        
                        if len(np.unique(y_binary_val)) > 1:
                            val_time_metrics[f'auc_{year}yr'] = roc_auc_score(y_binary_val, risk_scores_val)
                            val_time_metrics[f'auprc_{year}yr'] = average_precision_score(y_binary_val, risk_scores_val)
                        else:
                            val_time_metrics[f'auc_{year}yr'] = np.nan
                            val_time_metrics[f'auprc_{year}yr'] = np.nan
                        
                        logger.info(f"{year}yr - Train AUC: {train_time_metrics[f'auc_{year}yr']:.4f}, "
                                  f"AUPRC: {train_time_metrics[f'auprc_{year}yr']:.4f} | "
                                  f"Val AUC: {val_time_metrics[f'auc_{year}yr']:.4f}, "
                                  f"AUPRC: {val_time_metrics[f'auprc_{year}yr']:.4f}")
                    except Exception as e:
                        logger.warning(f"Could not compute {year}yr metrics: {e}")
                        train_time_metrics[f'auc_{year}yr'] = np.nan
                        train_time_metrics[f'auprc_{year}yr'] = np.nan
                        val_time_metrics[f'auc_{year}yr'] = np.nan
                        val_time_metrics[f'auprc_{year}yr'] = np.nan
                
                # For survival, keep C-index as main metric
                train_prauc = train_auc  # C-index for training
                val_prauc = val_auc  # C-index for validation
                best_auroc = val_auc  # C-index is the primary metric for survival
            elif algorithm == 'xgboost':
                logger.info("Performing hyperparameter grid search...")
                
                param_grid = {
                    'n_estimators': [50, 70],
                    'max_depth': [2, 3]
                }

                auc_scorer = make_scorer(roc_auc_score)

                xgb_base = XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='auc'
                )

                grid_search = GridSearchCV(
                    estimator=xgb_base,
                    param_grid=param_grid,
                    scoring=auc_scorer,
                    cv=10,
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                logger.info(f"Best parameters found (AUC: {grid_search.best_score_:.4f}): {best_params}")

            # Create and train the final XGBoost model
            logger.info("Training XGBoost model with parameters:")
            for param, value in best_params.items():
                logger.info(f"  {param}: {value}")

            xgb_model = XGBClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                    learning_rate=best_params.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1,
                eval_metric='auc'
            )

            logger.info(f"Training XGBoost on {len(X_train)} samples...")
            xgb_model.fit(X_train, y_train)
            logger.info("XGBoost training complete!")

            # Evaluate the model
            def evaluate_model(model, X, y, dataset_name):
                y_pred_proba = model.predict_proba(X)[:, 1]
                auc_score = roc_auc_score(y, y_pred_proba)
                prauc = average_precision_score(y, y_pred_proba)
                logger.info(f"{dataset_name} Set - AUC: {auc_score:.4f}, PRAUC: {prauc:.4f}")
                return auc_score, prauc

            train_auc, train_prauc = evaluate_model(xgb_model, X_train, y_train, "Training")
            val_auc, val_prauc = evaluate_model(xgb_model, X_val, y_val, "Validation")

            best_auroc = val_auc

            # Get feature importance (handle both sklearn and Booster models)
            if hasattr(xgb_model, 'feature_importances_'):
                # sklearn API (XGBClassifier)
                feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            else:
                # Booster API (xgb.train for survival)
                importance_dict = xgb_model.get_score(importance_type='weight')
                importances = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_cols))]
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importances
                }).sort_values('importance', ascending=False)

            logger.info("\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                logger.info(f"  {i}. {row['feature']}: {row['importance']:.4f}")

            # Save the model
            model_path = os.path.join(self.models_output_dir, f"{save_name}.pkl")
            config_path = os.path.join(self.models_output_dir, f"{save_name}_config.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': xgb_model,
                    'feature_cols': feature_cols,
                }, f)

            # Save configuration
            config = {
                "hyperparameters": best_params,
                "algorithm": algorithm,
                "metrics": {
                    "train_auc": train_auc,
                    "train_prauc": train_prauc,
                    "val_auc": val_auc,
                    "val_prauc": val_prauc
                },
                "feature_importance": feature_importance.to_dict(),
                "feature_cols": feature_cols
            }
            
            # Add time-dependent metrics for survival models
            if 'survival' in algorithm.lower():
                config["metrics"]["train_time_metrics"] = train_time_metrics
                config["metrics"]["val_time_metrics"] = val_time_metrics
            
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)

            if 'survival' in algorithm.lower():
                logger.info(f"XGBoost survival model training completed successfully. "
                          f"Validation C-index: {val_auc:.3f}, "
                          f"1yr AUC: {val_time_metrics.get('auc_1yr', np.nan):.3f}, "
                          f"1yr AUPRC: {val_time_metrics.get('auprc_1yr', np.nan):.3f}")
            else:
                logger.info(f"XGBoost model training completed successfully. Best Validation AUROC: {best_auroc:.3f} AUPRC: {val_prauc:.3f}")

            # Create metadata
            return Metadata.create_agent_output(
                status="success",
                dataset={
                    "train_data": {
                        "saved_path": train_data_path,
                        "description": "Training volume features",
                        "configuration": {"dataset_len": len(X_train), "num_features": len(feature_cols)}
                    },
                    "validation_data": {
                        "saved_path": validation_data_path,
                        "description": "Validation volume features",
                        "configuration": {"dataset_len": len(X_val), "num_features": len(feature_cols)}
                    },
                    "model_performance": {
                        "saved_path": None,
                        "description": "Model performance metrics",
                        "configuration": {
                            "val_auc": val_auc,
                            "val_prauc": val_prauc
                        }
                    }
                },
                model={
                    "xgb_model": {
                        "saved_path": model_path,
                        "description": f"XGBoost model trained on brain volume features for {'survival analysis' if 'survival' in algorithm.lower() else 'binary classification'}",
                        "configuration": {
                            "best_auroc": best_auroc,
                            "best_prauc": val_prauc,
                            "architecture": "XGBoost" + (" Survival" if 'survival' in algorithm.lower() else ""),
                            "num_features": len(feature_cols),
                            "training_hyperparameters": best_params,
                            "top_features": feature_importance.head(10).to_dict('records'),
                            **({'time_metrics': val_time_metrics} if 'survival' in algorithm.lower() else {})
                        }
                    }
                },
                cache_directory=CACHE_DIR,
                agent_name="image_agent"
            )

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                filename, lineno, func, text = tb[-1]
                logger.error(
                    f"Error during XGBoost pipeline execution: {str(e)}\n"
                    f"Occurred in file: {filename}, line {lineno}, in {func}\n"
                    f"Code: {text}"
                )
            else:
                logger.error(f"Error during XGBoost pipeline execution: {str(e)}")

            return Metadata.create_agent_output(
                status="error",
                dataset={
                    "error": {
                        "saved_path": None,
                        "description": str(e),
                        "configuration": {
                            "train_data_path": train_data_path,
                            "validation_data_path": validation_data_path,
                            "algorithm": algorithm
                        }
                    }
                },
                model={
                    save_name: {
                        "saved_path": "",
                        "description": f"Error during model training",
                        "configuration": {"error_message": str(e)}
                    }
                },
                cache_directory=CACHE_DIR,
                agent_name="image_agent",
                status_description=f"Error during model training: {str(e)}"
            )


if __name__ == "__main__":
    # Test the XGBoost volume trainer tool
    tool = Image_Model_Trainer_Tool()

    # Example with volume CSV files
    result = tool.execute(
        train_data_path='path/to/X_mri_volume_train.csv',
        train_labels_path='path/to/y_mri_volume_train.csv',
        validation_data_path='path/to/X_mri_volume_val.csv',
        validation_labels_path='path/to/y_mri_volume_val.csv',
        algorithm='xgboost',  # or 'survival' for survival model
        save_name="test_xgb_volume_model"
    )

    # Extract data from result
    metadata_dict = result.get_metadata_info()

    if metadata_dict["status"] == "success":
        print("✅ XGBoost volume pipeline completed successfully!")
        print(f"Model saved to: {metadata_dict['model']['xgb_model']['saved_path']}")
        print(f"Validation AUROC: {metadata_dict['model']['xgb_model']['configuration']['best_auroc']:.4f}")

        print("\nDataset files:")
        for name, info in metadata_dict['dataset'].items():
            print(f"  {name}: {info['saved_path']}")

        print(f"\nTop Features: {metadata_dict['model']['xgb_model']['configuration']['top_features']}")
    else:
        print(f"❌ XGBoost volume pipeline failed: see logs for details.")
