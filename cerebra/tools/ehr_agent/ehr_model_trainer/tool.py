import pandas as pd
import numpy as np
import joblib
import logging
import os
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import load_npz, vstack
from cerebra.tools.base import BaseTool
from cerebra.utils.metadata import Metadata
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from xgboost import XGBClassifier
import dotenv

dotenv.load_dotenv()
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")

logger = logging.getLogger(__name__)



class EHR_Model_Trainer_Tool(BaseTool):
    """
    A tool for training machine learning models (Random Forest, XGBoost, Random Survival Forest)
    on EHR data with hyperparameter tuning using separate train/validation sets.
    Compatible with the new lightweight_agent architecture.
    """

    def __init__(self):
        super().__init__()
        self.model_output_dir = os.path.join(CACHE_DIR, "ehr_agent", "models")
        os.makedirs(self.model_output_dir, exist_ok=True)
        self.set_metadata(
            tool_name="EHR_Model_Trainer_Tool",
            tool_description="Preprocess the EHR data and train machine learning models (Random Forest, XGBoost, Random Survival Forest) on processed data.",
            tool_version="2.1.0",
            input_types = {
                "train_data_path": "str - Path to the training feature file (.npz or .pkl)",
                "train_labels_path": "str - Path to the training labels file (.npy or .pkl). For classification: array of labels. For survival: array of (time, event) tuples",
                "validation_data_path": "str - Path to the validation feature file (.npz or .pkl)",
                "validation_labels_path": "str - Path to the validation labels file (.npy or .pkl). For classification: array of labels. For survival: array of (time, event) tuples",
                "algorithm": "str - Model algorithm to train with ('random_forest', 'xgboost', or 'cox')",
            },
            output_type="Metadata - Metadata object containing input dataset, input model, output dataset, output model, and other configurations",

            demo_commands = [
                {
                    "command":
                        "result = tool.execute(\n"
                        "    train_data_path=dataset['train_data']['saved_path'],\n"
                        "    train_labels_path=dataset['train_labels']['saved_path'],\n"
                        "    validation_data_path=dataset['validation_data']['saved_path'],\n"
                        "    validation_labels_path=dataset['validation_labels']['saved_path'],\n"
                        "    algorithm='xgboost'\n"
                        ")",
                    "description": "Process EHR data and train a XGBoost model using paths from the Metadata output."
                }
            ],
            user_metadata={
                "limitations": [
                    "Requires separate train and validation datasets",
                    "Limited to binary classification tasks",
                    "Expects data in the format of the data dictionary"
                ],
                "best_practices": [
                    "Use Dataset objects for better integration with lightweight agents",
                    "Normalize features before training",
                    "Check for class imbalance in target labels",
                    "dataset['train_data'] and dataset['validation_data'] are expected to be the data path"
                ]
            },
            evaluation_criteria={
                "technical_success": "Did the tool execute without errors?",
                "performance_evaluation": "Is the model performance satisfactory?",
                "improvement_opportunities": "What specific parameters should be adjusted?",
                "efficiency": "It is not necessary to achieve the perfect performance, the task should be considered completed if the performance is satisfactory to save time and resources."
            }
        )


    def load_data_from_dataset(self, train_data, train_labels, validation_data, validation_labels):
        """
        Load training and validation data from Dataset object.

        Args:
            train_data (list of sparse matrices or single sparse matrix/array): Training data
            train_labels (list of labels or single array): Training labels
            validation_data (list of sparse matrices or single sparse matrix/array): Validation data
            validation_labels (list of labels or single array): Validation labels

        Returns:
            tuple: (X_train, y_train, X_val, y_val) as numpy arrays
        """
        # Handle training data - concatenate if it's a list of sparse matrices
        y_train = []
        if isinstance(train_data, list):
            temporal_processed_data = []
            for i, data in enumerate(train_data):
                if data.shape[0] > 0:
                    temporal_processed_data.append(data.max(axis=0))
                else:
                    from scipy.sparse import csr_matrix
                    temporal_processed_data.append(csr_matrix((1, data.shape[1])))
                y_train.extend([train_labels[i]]*temporal_processed_data[-1].shape[0])
            train_data = temporal_processed_data

            logger.info(f"Concatenating {len(train_data)} training sparse matrices")
            X_train = vstack(train_data)
            y_train = np.array(y_train)
        else:
            X_train = train_data

        # Handle validation data - concatenate if it's a list of sparse matrices
        if isinstance(validation_data, list):
            temporal_processed_data = []
            y_val = []
            for i, data in enumerate(validation_data):
                if data.shape[0] > 0:
                    temporal_processed_data.append(data.max(axis=0))
                else:
                    from scipy.sparse import csr_matrix
                    temporal_processed_data.append(csr_matrix((1, data.shape[1])))
                y_val.extend([validation_labels[i]]*temporal_processed_data[-1].shape[0])
            validation_data = temporal_processed_data

            logger.info(f"Concatenating {len(validation_data)} validation sparse matrices")
            X_val = vstack(validation_data)
            y_val = np.array(y_val)
        else:
            X_val = validation_data

        y_train = np.array(train_labels) if isinstance(train_labels, list) else train_labels
        y_val = np.array(validation_labels)
        assert X_train.shape[0] == len(y_train), "Number of training samples and labels must match"
        assert X_val.shape[0] == len(y_val), "Number of validation samples and labels must match"

        logger.info(f"Loaded from Dataset - Train: {X_train.shape}, Val: {X_val.shape}")
        return X_train, y_train, X_val, y_val

    def load_survival_data_from_dataset(self, train_data, train_labels, validation_data, validation_labels):
        """
        Load training and validation data for survival analysis.
        Labels should be array/list of (time, event) tuples.
        Returns structured arrays compatible with scikit-survival.

        Args:
            train_data: Training features
            train_labels: Training labels as (time, event) tuples
            validation_data: Validation features
            validation_labels: Validation labels as (time, event) tuples

        Returns:
            tuple: (X_train, y_train, X_val, y_val) where y are structured arrays
        """
        # Handle training data
        if isinstance(train_data, list):
            temporal_processed_data = []
            t_train = []
            e_train = []
            for i, data in enumerate(train_data):
                if data.shape[0] > 0:
                    temporal_processed_data.append(data.max(axis=0))
                else:
                    from scipy.sparse import csr_matrix
                    temporal_processed_data.append(csr_matrix((1, data.shape[1])))
                n_samples = temporal_processed_data[-1].shape[0]
                # Unpack (time, event) tuple
                time_val, event_val = train_labels[i]
                t_train.extend([time_val] * n_samples)
                e_train.extend([event_val] * n_samples)
            
            X_train = vstack(temporal_processed_data)
            t_train = np.array(t_train)
            e_train = np.array(e_train, dtype=bool)
        else:
            X_train = train_data
            # Unpack array of (time, event) tuples
            train_labels_array = np.array(train_labels)
            if train_labels_array.ndim == 2:
                t_train = train_labels_array[:, 0].astype(float)
                e_train = train_labels_array[:, 1].astype(bool)
            else:
                t_train = np.array([t for t, e in train_labels], dtype=float)
                e_train = np.array([e for t, e in train_labels], dtype=bool)

        # Handle validation data
        if isinstance(validation_data, list):
            temporal_processed_data = []
            t_val = []
            e_val = []
            for i, data in enumerate(validation_data):
                if data.shape[0] > 0:
                    temporal_processed_data.append(data.max(axis=0))
                else:
                    from scipy.sparse import csr_matrix
                    temporal_processed_data.append(csr_matrix((1, data.shape[1])))
                n_samples = temporal_processed_data[-1].shape[0]
                # Unpack (time, event) tuple
                time_val, event_val = validation_labels[i]
                t_val.extend([time_val] * n_samples)
                e_val.extend([event_val] * n_samples)
            
            X_val = vstack(temporal_processed_data)
            t_val = np.array(t_val)
            e_val = np.array(e_val, dtype=bool)
        else:
            X_val = validation_data
            # Unpack array of (time, event) tuples
            val_labels_array = np.array(validation_labels)
            if val_labels_array.ndim == 2:
                t_val = val_labels_array[:, 0].astype(float)
                e_val = val_labels_array[:, 1].astype(bool)
            else:
                t_val = np.array([t for t, e in validation_labels], dtype=float)
                e_val = np.array([e for t, e in validation_labels], dtype=bool)

        # Create structured arrays for scikit-survival
        y_train = np.empty(len(t_train), dtype=[('event', bool), ('time', float)])
        y_train['event'] = e_train
        y_train['time'] = t_train

        y_val = np.empty(len(t_val), dtype=[('event', bool), ('time', float)])
        y_val['event'] = e_val
        y_val['time'] = t_val

        assert X_train.shape[0] == len(y_train), "Training samples and labels must match"
        assert X_val.shape[0] == len(y_val), "Validation samples and labels must match"

        logger.info(f"Loaded survival data - Train: {X_train.shape}, Val: {X_val.shape}")
        return X_train, y_train, X_val, y_val

    def train_model(self, X_train, y_train, X_val, y_val, algorithm='random_forest', random_state=42):
        """
        Train a classification model on training data and validate on validation data.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training target labels.
            X_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation target labels.
            algorithm (str): Model to train ('random_forest' or 'logistic_regression').
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: (trained_model, validation_metrics, training_info)
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        results = []
        # Grid search over hyperparameters for Random Forest
        if algorithm == 'random_forest':
            param_grid = {'n_estimators': [80, 100, 120], 'max_depth': [5, 10, 15]}
            total = len(param_grid['n_estimators']) * len(param_grid['max_depth'])
            pbar = tqdm(total=total, desc="Training Random Forest")

            for n in param_grid['n_estimators']:
                for d in param_grid['max_depth']:
                    model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=random_state, class_weight='balanced_subsample')
                    model.fit(X_train, y_train)
                    pred = model.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, pred)
                    prauc = average_precision_score(y_val, pred)

                    logger.info(f"RF n={n}, depth={d} | val_auc={auc:.3f} | val_prauc={prauc:.3f}")

                    results.append({
                        'model': model,
                        'AUROC': auc,
                        'PRAUC': prauc,
                        'params': {'n_estimators': n, 'max_depth': d}
                    })
                    pbar.update(1)
            pbar.close()

        # Grid search over hyperparameter C for Logistic Regression
        elif algorithm == 'xgboost':
            param_grid = {
            'n_estimators': [80, 100, 120],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.2]
            }
        
            # Calculate scale_pos_weight for class imbalance
            # scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            pbar = tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']), desc="Training XGBoost")
            
            for n in param_grid['n_estimators']:
                for d in param_grid['max_depth']:
                    for lr in param_grid['learning_rate']:
                        model = XGBClassifier(
                            n_estimators=n,
                            max_depth=d,
                            learning_rate=lr,
                            # scale_pos_weight=scale_pos_weight,
                            random_state=42,
                            eval_metric='logloss',
                            n_jobs=-1
                        )
                        model.fit(X_train, y_train)
                        pred = model.predict_proba(X_val)[:, 1]
                        auc = roc_auc_score(y_val, pred)
                        prauc = average_precision_score(y_val, pred)

                        logger.info(f"XGB n={n}, depth={d}, learning_rate={lr} | val_auc={auc:.3f} | val_prauc={prauc:.3f}")

                        results.append({
                            'model': model,
                            'AUROC': auc,
                            'PRAUC': prauc,
                            'params': {'n_estimators': n, 'max_depth': d, 'learning_rate': lr}
                        })
                        pbar.update(1)
            pbar.close()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'random_forest' or 'xgboost'")

        # Find best model based on validation AUROC
        best_result = max(results, key=lambda x: x['AUROC'])
        best_model = best_result['model']

        training_info = {
            'algorithm': algorithm,
            'best_params': best_result['params'],
            'n_train_samples': X_train.shape[0],
            'n_val_samples': X_val.shape[0],
            'n_features': X_train.shape[1],
            'random_state': random_state,
            'all_results': results
        }

        return best_model, best_result, training_info

    def train_cox_model(self, X_train, y_train, X_val, y_val, random_state=42):
        """
        Train a Random Survival Forest model using scikit-survival.
        RandomSurvivalForest supports sparse matrices directly.

        Args:
            X_train: Training feature matrix (sparse or dense)
            y_train: Training labels as structured array with 'event' and 'time' fields
            X_val: Validation feature matrix (sparse or dense)
            y_val: Validation labels as structured array with 'event' and 'time' fields
            random_state: Random seed

        Returns:
            tuple: (trained_model, validation_metrics, training_info)
        """
        results = []
        # Grid search over hyperparameters for Random Survival Forest
        param_grid = {
            'n_estimators': [80, 100, 120, 140],
            'max_depth': [6, 7, 8, 9]
        }
        total = len(param_grid['n_estimators']) * len(param_grid['max_depth'])
        pbar = tqdm(total=total, desc="Training Random Survival Forest")

        for n in param_grid['n_estimators']:
            for d in param_grid['max_depth']:
                    try:
                        rsf = RandomSurvivalForest(
                            n_estimators=n,
                            max_depth=d,
                            random_state=random_state,
                            n_jobs=-1  # Use all available cores
                        )
                        rsf.fit(X_train, y_train)

                        # Calculate C-index on validation set
                        pred_risk = rsf.predict(X_val)

                        c_result = concordance_index_censored(y_val['event'], y_val['time'], pred_risk)
                        c_index = c_result[0]
                        
                        logger.info(f"RSF n={n}, depth={d} | val_c_index={c_index:.3f}")
                        print(f"RSF n={n}, depth={d} | val_c_index={c_index:.3f}")
                        
                        results.append({
                            'model': rsf,
                            'C_index': c_index,
                            'params': {
                                'n_estimators': n,
                                'max_depth': d,
                            }
                        })
                    except Exception as e:
                        logger.warning(f"RSF model failed with n={n}, depth={d}: {str(e)}")
                        continue
                    
                    pbar.update(1)
        pbar.close()

        if not results:
            raise ValueError("All Random Survival Forest configurations failed")

        best_result = max(results, key=lambda x: x['C_index'])
        best_model = best_result['model']

        training_info = {
            'algorithm': 'random_survival_forest',
            'best_params': best_result['params'],
            'n_train_samples': X_train.shape[0],
            'n_val_samples': X_val.shape[0],
            'n_features': X_train.shape[1],
            'random_state': random_state,
            'all_results': results
        }

        return best_model, best_result, training_info

    def save_model(self, model, model_path):
        """
        Save the trained model to disk.

        Args:
            model: Trained model to save.
            model_path (str): model_path

        Returns:
            str: Path to the saved model file.
        """
        try:
            joblib.dump(model, model_path)
            logger.info(f"Model saved to: {model_path}")
            return model_path
        except:
            return None

    def execute(self,
                train_data_path=None,
                train_labels_path=None,
                validation_data_path=None,
                validation_labels_path=None,
                algorithm='xgboost',
                model_name=None,
                **kwargs):
        """
        Execute the ML model training tool.

        Args:
            train_data_path: Path to training features
            train_labels_path: Path to training labels (scalars for classification, (time,event) tuples for survival)
            validation_data_path: Path to validation features
            validation_labels_path: Path to validation labels (scalars for classification, (time,event) tuples for survival)
            algorithm: Model algorithm ('random_forest', 'xgboost', or 'cox')
            model_name: Model name to save as
            **kwargs: Additional arguments

        Returns:
            Metadata: Metadata object containing model and performance metrics
        """
        random_state = 42
        if model_name is None:
            model_name = "trained_model"

        try:
            def try_load_file(value):
                if isinstance(value, str):
                    if value.endswith(".npz"):
                        return load_npz(value)
                    elif value.endswith(".npy"):
                        return np.load(value, allow_pickle=True)  # allow_pickle for tuples
                    elif value.endswith(".pkl"):
                        with open(value, 'rb') as f:
                            return pickle.load(f)
                return value

            train_data = try_load_file(train_data_path)
            train_labels = try_load_file(train_labels_path)
            validation_data = try_load_file(validation_data_path)
            validation_labels = try_load_file(validation_labels_path)
   
            
            if train_data is None or train_labels is None or validation_data is None or validation_labels is None:
                raise ValueError("All training and validation data paths must be provided.")

            # Determine if this is a survival task based on algorithm
            if algorithm == 'random_survival_forest':
                logger.info("Loading and processing survival data for Cox model")
                X_train, y_train, X_val, y_val = self.load_survival_data_from_dataset(
                    train_data, train_labels, validation_data, validation_labels
                )
                # Check if model already exists
                if os.path.exists(os.path.join(self.model_output_dir, f"{model_name}.joblib")) and \
                   os.path.exists(os.path.join(self.model_output_dir, f"{model_name}_config.pkl")):
                    logger.info(f"Model {model_name} already exists. Skipping training.")
                    trained_model = joblib.load(os.path.join(self.model_output_dir, f"{model_name}.joblib"))
                    best_val_metrics = pickle.load(open(os.path.join(self.model_output_dir, f"{model_name}_config.pkl"), "rb"))
                    model_path = os.path.join(self.model_output_dir, f"{model_name}.joblib")
                else:
                    # Train Random Survival Forest model
                    trained_model, best_val_metrics, _ = self.train_cox_model(
                        X_train, y_train, X_val, y_val, random_state
                    )
                    # Save model (can use joblib like other sklearn models!)
                    model_path = os.path.join(self.model_output_dir, f"{model_name}.joblib")
                    joblib.dump(trained_model, model_path)
                    pickle.dump(best_val_metrics, open(os.path.join(self.model_output_dir, f"{model_name}_config.pkl"), "wb"))

                logger.info(f"Random Survival Forest training completed successfully. Validation C-index: {best_val_metrics['C_index']:.3f}")

                return Metadata.create_agent_output(
                    status="success",
                    dataset={
                        "train_features": {
                            "saved_path": train_data_path,
                            "description": "Training feature data",
                            "configuration": {"dataset_len": len(y_train)}
                        },
                        "train_labels": {
                            "saved_path": train_labels_path,
                            "description": "Training time-to-event and event indicators as (time, event) tuples",
                            "configuration": {"dataset_len": len(y_train)}
                        },
                        "validation_features": {
                            "saved_path": validation_data_path,
                            "description": "Validation feature data",
                            "configuration": {"dataset_len": len(y_val)}
                        },
                        "validation_labels": {
                            "saved_path": validation_labels_path,
                            "description": "Validation time-to-event and event indicators as (time, event) tuples",
                            "configuration": {"dataset_len": len(y_val)}
                        },
                        "model_performance": {
                            "saved_path": None,
                            "description": "Validation C-index for the trained survival model",
                            "configuration": {"model_performance": best_val_metrics['C_index']}
                        }
                    },
                    model={
                        "trained_model": {
                            "saved_path": model_path,
                            "description": f"Trained {algorithm} model stored as joblib",
                            "configuration": {
                                "validation_C_index": best_val_metrics['C_index'],
                                "best_parameters": best_val_metrics['params']
                            }
                        }
                    },
                    cache_directory=os.path.join(CACHE_DIR, "ehr_agent", "ml_models"),
                    agent_name="ehr_agent"
                )

            else:
                # Classification task - use existing code
                logger.info("Loading and processing training data")
                X_train, y_train, X_val, y_val = self.load_data_from_dataset(
                    train_data, train_labels, validation_data, validation_labels
                )

                # Train the model using train and validation data
                trained_model, best_val_metrics, _ = self.train_model(
                    X_train, y_train, X_val, y_val, algorithm, random_state
                )
                # Save model
                model_path = self.save_model(trained_model, os.path.join(self.model_output_dir, f"{model_name}.joblib"))
                pickle.dump(best_val_metrics, open(os.path.join(self.model_output_dir, f"{model_name}_config.pkl"), "wb"))

                # Create output data for Dataset
                output_data = {
                    "trained_model_path": model_path,
                    "algorithm": algorithm,
                    "validation_AUROC": best_val_metrics['AUROC'],
                    "best_parameters": best_val_metrics['params'],
                    "training_metadata": {
                        "n_train_samples": X_train.shape[0],
                        "n_val_samples": X_val.shape[0],
                        "n_features": X_train.shape[1],
                        "random_state": random_state,
                        "algorithm": algorithm,
                    },
                    "status": "success"
                }

                feature_descriptions = {
                    "trained_model_path": f"Path to the saved trained model file (joblib format): {model_path}",
                    "algorithm": "Machine learning algorithm used for training",
                    "validation_AUROC": "Best validation AUROC achieved",
                    "best_parameters": "Best hyperparameters found during training",
                    "training_metadata": "Metadata about the training process",
                    "status": "Training execution status"
                }

                # Return as Dataset object for compatibility with lightweight agent
                model_name_key = model_name or "EHR_trained_model"
                cache_dir = os.path.join(self.output_dir or CACHE_DIR, "ehr_agent", "ml_models")

                logger.info(f"Model training completed successfully. Validation AUROC: {best_val_metrics['AUROC']:.3f}. AUPRC: {best_val_metrics['PRAUC']:.3f}")
                # Use actual saved file paths if available, else None
                train_data_entry = {
                    "saved_path": train_data_path,
                    "description": "Training feature data used to train the model",
                    "configuration": {"dataset_len": len(y_train)}
                }

                train_labels_entry = {
                    "saved_path": train_labels_path,
                    "description": "Training labels used to train the model",
                    "configuration": {"dataset_len": len(y_train)}
                }

                val_data_entry = {
                    "saved_path": validation_data_path,
                    "description": "Validation feature data used to validate the model",
                    "configuration": {"dataset_len": len(y_val)}
                }

                val_labels_entry = {
                    "saved_path": validation_labels_path,
                    "description": "Validation labels used to validate the model",
                    "configuration": {"dataset_len": len(y_val)}
                }


                model_entry = {
                    "saved_path": model_path,
                    "description": f"Trained {algorithm} model stored as joblib",
                    "configuration": {
                        "validation_AUROC": best_val_metrics['AUROC'],
                        "best_parameters": best_val_metrics['params']
                    }
                }

                return Metadata.create_agent_output(
                    status="success",
                    dataset={
                        "train_features": train_data_entry,
                        "train_labels": train_labels_entry,
                        "validation_features": val_data_entry,
                        "validation_labels": val_labels_entry,
                        "model_performance": {
                            "saved_path": None,
                            "description": "Validation AUC-ROC for the trained model",
                            "configuration": {"model_performance": best_val_metrics['AUROC']}
                        }
                    },
                    model={
                        "trained_model": model_entry
                    },
                    cache_directory=os.path.join(self.output_dir or CACHE_DIR, "ehr_agent", "ml_models"),
                    agent_name="ehr_agent"
                )


        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Error during model training: {str(e)}")
            logger.error(traceback.format_exc())

            # Return error as Dataset for consistency
            error_data = {
                "status": "error",
                "error_message": str(e),
                "algorithm": algorithm,
                "parameters": {
                    "random_state": random_state,
                    "model_name": model_name
                }
            }

            error_descriptions = {
                "status": "Training execution status (error)",
                "error_message": "Description of the error that occurred",
                "algorithm": "Algorithm that was attempted",
                "parameters": "Parameters used when error occurred"
            }

            cache_dir = os.path.join(self.output_dir or CACHE_DIR, "ehr_agent")

            result_metadata = Metadata.create_agent_output(
                status="error",
                dataset={},
                model={
                    model_name: {
                        "saved_path": "",
                        "description": f"Error during {algorithm} model training",
                        "configuration": error_data
                    }
                },
                cache_directory=cache_dir,
                status_description=f"Error during {algorithm} model training"
            )
            return result_metadata


if __name__ == "__main__":
    # Test the tool
    print("Testing EHR_Model_Trainer_Tool...")

    # Use actual file paths
    from cerebra.agents.data_agent import DataAgent
    year = 1
    data_agent = DataAgent()
    time_to_event = True
    balanced = False
    dedup = True
    input_metadata = data_agent.run(f"Load initial data for ehr_agent", agent_name='ehr_agent', patient_id=None, year=year, time_to_event=time_to_event, balanced=balanced, dedup=dedup)
    train_data_path = input_metadata.get_metadata_info()['dataset']['train_data']['saved_path']
    train_labels_path = input_metadata.get_metadata_info()['dataset']['train_labels']['saved_path']
    validation_data_path = input_metadata.get_metadata_info()['dataset']['validation_data']['saved_path']
    validation_labels_path = input_metadata.get_metadata_info()['dataset']['validation_labels']['saved_path']


    # Check if files exist
    if not os.path.exists(train_data_path):
        print(f"❌ Train data file not found: {train_data_path}")
        exit(1)

    if not os.path.exists(train_labels_path):
        print(f"❌ Train labels file not found: {train_labels_path}")
        exit(1)

    if not os.path.exists(validation_data_path):
        print(f"❌ Validation data file not found: {validation_data_path}")
        exit(1)

    if not os.path.exists(validation_labels_path):
        print(f"❌ Validation labels file not found: {validation_labels_path}")
        exit(1)

    print("✅ Files found!")

    # Initialize the tool
    tool = EHR_Model_Trainer_Tool()

    # Test Random Forest training with file paths
    print("\n=== Testing Random Forest Training (File Paths) ===")

    result = tool.execute(
        train_data_path=train_data_path,
        train_labels_path=train_labels_path,
        validation_data_path=validation_data_path,
        validation_labels_path=validation_labels_path,
        algorithm='xgboost' if not time_to_event else 'cox',
        model_name=f'trained_ehr_model_time_to_event' if time_to_event else f'trained_ehr_model_{year}yr'
    )

    # Extract data from Dataset result
    result_data = result.get_metadata_info()["dataset"]["model_performance"]

    if result.get_metadata_info()['status'] == 'success':
        print("✅ Random Forest training successful!")
        if time_to_event:
            print(f"Best validation C-index: {result_data['configuration']['model_performance']:.3f}")
        else:
            print(f"Best validation AUROC: {result_data['configuration']['model_performance']:.3f}")
    else:
        print(f"❌ Random Forest training failed: {result_data.get('error_message', 'Unknown error')}")

    # # Test Logistic Regression training with file paths
    # print("\n=== Testing Logistic Regression Training (File Paths) ===")
    # try:
    #     result = tool.execute(
    #         train_data_path=train_data_path,
    #         train_labels_path=train_labels_path,
    #         validation_data_path=validation_data_path,
    #         validation_labels_path=validation_labels_path,
    #         algorithm='logistic_regression' if not time_to_event else 'cox',
    #         model_name=f'trained_ehr_model_lr_time_to_event' if time_to_event else f'trained_ehr_model_lr_{year}yr'
    #     )

    #     # Extract data from Dataset result
    #     result_data = result.get_metadata_info()["dataset"]["model_performance"]

    #     if result.get_metadata_info()['status'] == 'success':
    #         print("✅ Logistic Regression training successful!")
    #         print(f"Best validation AUROC: {result_data['configuration']['model_performance']:.3f}")
    #     else:
    #         print(f"❌ Logistic Regression training failed: {result_data.get('error_message', 'Unknown error')}")

    # except Exception as e:
    #     print(f"❌ Error during Logistic Regression training: {str(e)}")

    # print("\n=== Test Complete ===")