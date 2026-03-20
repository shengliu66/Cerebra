import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from cerebra.utils.metadata import Metadata
import uuid
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from cerebra.tools.base import BaseTool
from cerebra.utils.log_utils import setup_logger
# Configure logging to save to the tool's cache directory
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
CACHE_DIR = os.environ.get("CEREBRA_CACHE_DIR", "cerebra_cache")


log_dir = os.path.join(CACHE_DIR, "note_agent", "logs")
# Setup logger
logger = setup_logger(log_dir)

# Import the models from the note_agent models directory
from cerebra.tools.note_agent.models.sentence_attention import SentenceAttentionBERT

import traceback
from sklearn.metrics import roc_auc_score, average_precision_score
from pycox.models.loss import CoxPHLoss
from sksurv.metrics import concordance_index_censored

class NoteDataset(TorchDataset):
    """Dataset class for loading note embeddings and labels."""
    def __init__(self, feature_path, label_path, max_len=30):
        with open(feature_path, 'rb') as f:
            self.note_embeddings = pickle.load(f)
        self.labels = np.load(label_path)
        
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def get_max_len(self, note_embeddings):
        return max([len(note) for note in note_embeddings])

    def __getitem__(self, idx):
        note = self.note_embeddings[idx]  # List of [768] vectors
        label = self.labels[idx]
        # max_len = self.get_max_len(note)
        max_len = self.max_len
        # Pad/truncate to max_len
        if len(note) >= max_len:
            note = note[-max_len:]
        else:
            pad = [np.zeros(note[0].shape[0])] * (max_len - len(note))
            note += pad
        return torch.tensor(note, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class NoteCoxDataset(TorchDataset):
    """Dataset class for loading note embeddings with survival labels (time, event)."""
    def __init__(self, feature_path, label_path, max_len=30):
        with open(feature_path, 'rb') as f:
            self.note_embeddings = pickle.load(f)
        labels = np.load(label_path, allow_pickle=True)
        # Handle (time, event) tuples
        if labels.ndim == 2:
            self.times = labels[:, 0].astype(np.float32)
            self.events = labels[:, 1].astype(np.float32)
        else:
            self.times = np.array([t for t, e in labels], dtype=np.float32)
            self.events = np.array([e for t, e in labels], dtype=np.float32)
        
        self.max_len = max_len

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        note = self.note_embeddings[idx]
        time = self.times[idx]
        event = self.events[idx]
        
        # Pad/truncate to max_len
        if len(note) >= self.max_len:
            note = note[-self.max_len:]
        else:
            pad = [np.zeros(note[0].shape[0])] * (self.max_len - len(note))
            note += pad
        
        return torch.tensor(note, dtype=torch.float32), torch.tensor(time, dtype=torch.float32), torch.tensor(event, dtype=torch.float32)


class Note_Model_Trainer_Tool(BaseTool):
    """
    An end-to-end tool that extracts features from clinical notes and trains a classification or survival model.
    First extracts sentence-level embeddings from clinical notes, then trains a model on those features.
    Supports both binary classification (default) and survival analysis using Cox proportional hazards.
    """

    def __init__(self):
        super().__init__()

        # Output directories
        self.features_output_dir = os.path.join(CACHE_DIR, "note_agent", "features")
        self.models_output_dir = os.path.join(CACHE_DIR, "note_agent", "models")
        os.makedirs(self.features_output_dir, exist_ok=True)
        os.makedirs(self.models_output_dir, exist_ok=True)

        self.set_metadata(
            tool_name="Note_Model_Trainer_Tool",
            tool_description="End-to-end tool that trains a classification or survival model (without performing prediction). Supports binary classification and Cox proportional hazards for survival analysis.",
            tool_version="1.2.0",
            input_types={
                "train_data_path(required)": "str - Path to pickle file containing training clinical notes",
                "train_labels_path(required)": "str - Path to numpy file containing training labels",
                "validation_data_path(required)": "str - Path to pickle file containing validation clinical notes",
                "validation_labels_path(required)": "str - Path to numpy file containing validation labels",
                "test_data_path(required)": "str - Path to pickle file containing test clinical notes",
                "test_labels_path(required)": "str - Path to numpy file containing test labels",
                "embedding_model_name(required)": "str - Name of the embedding model to use (default: 'Qwen/Qwen3-Embedding-0.6B')",
                "epochs": "int - Number of training epochs (default: 40)",
                "batch_size": "int - Training batch size (default: 256)",
                "learning_rate": "float - Learning rate for training (default: 1e-4)",
                "max_len": "int - Maximum number of sentences per note (default: 128)",
                "save_name": "str - Name for saving the trained model (default: 'note_trained_model')",
                "year": "int - Year of prediction (if diagnosis or survival, set to 0)",
                "task_type": "str - Type of task: 'classification_prediction' or 'classification_diagnosis' or 'survival'"
            },
            output_type="Metadata - Metadata object containing trained model path and validation performance",
            demo_commands=[
                {
                        "command": (
                        "result = tool.execute(\n"
                        "    train_data_path=dataset['train_data']['saved_path'],\n"
                        "    train_labels_path=dataset['train_labels']['saved_path'],\n"
                        "    validation_data_path=dataset['validation_data']['saved_path'],\n"
                        "    validation_labels_path=dataset['validation_labels']['saved_path'],\n"
                        "    test_data_path=dataset['test_data']['saved_path'],\n"
                        "    test_labels_path=dataset['test_labels']['saved_path'],\n"
                        "    embedding_model_name='Qwen/Qwen3-Embedding-0.6B',\n"
                        "    task_type='survival' if time_to_event else 'classification',\n"
                        "    save_name='note_trained_model' if time_to_event else 'note_trained_model',\n"
                        "    year=0\n"
                        ")"
                    ),
                    "description": "Train survival model using extracted note embeddings"
                },
                {
                        "command": (
                        "result = tool.execute(\n"
                        "    train_data_path=dataset['train_data']['saved_path'],\n"
                        "    train_labels_path=dataset['train_labels']['saved_path'],\n"
                        "    validation_data_path=dataset['validation_data']['saved_path'],\n"
                        "    validation_labels_path=dataset['validation_labels']['saved_path'],\n"
                        "    test_data_path=dataset['test_data']['saved_path'],\n"
                        "    test_labels_path=dataset['test_labels']['saved_path'],\n"
                        "    embedding_model_name='Qwen/Qwen3-Embedding-0.6B',\n"
                        "    task_type='classification_prediction',\n"
                        "    save_name='note_trained_model',\n"
                        "    year=3\n"
                        ")"
                    ),
                    "description": "Train 3 year risk prediction model using extracted note embeddings"
                },
            ],
            user_metadata={
                "limitations": [
                    "Only supports fixed transformer models (e.g., RoBERTa)",
                    "Requires all three splits to be provided",
                    "Only supports binary classification and survival analysis (Cox)",
                    "Cannot perform prediction on the test data"
                ],
                "best_practices": [
                    "This tool is often used together with the note_inference tool to perform prediction on the test data.",
                    "Use the embedding_model_name parameter to specify the embedding model to use",
                    "Set task_type='survival' for Cox survival or 'classification' for binary classification",
                    "For survival tasks, labels should be (time, event) tuples",
                    "Use consistent sentence segmentation across splits",
                    "Monitor validation AUC (classification) or C-index (survival) during training",
                    "Use higher batch sizes if GPU permits",
                    "Ensure consistent label encoding across splits"
                ]
            },
            evaluation_criteria={
                "technical_success": "Did the tool execute without errors?",
                "performance_evaluation": "Is the model performance satisfactory?",
                "improvement_opportunities": "What specific parameters should be adjusted?",
                "efficiency": "It is not necessary to achieve the perfect performance, the task should be considered completed if the performance is satisfactory to save time and resources."
            }
        )

    def _embed_sentences(self, sentences: List[str]) -> List[np.ndarray]:
        """Embed a list of sentences using the transformer model."""

        if self.embedding_model_name == "Qwen/Qwen2.5-0.5B" or self.embedding_model_name == "roberta-base":
            embeddings = []
            for sent in sentences:
                if not sent or not sent.strip():
                    # Skip empty sentences
                    continue

                inputs = self.tokenizer(sent, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    if self.embedding_model_name == "roberta-base":
                        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                    elif self.embedding_model_name in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B"]:
                        cls_embedding = outputs.last_hidden_state[:, -1, :].squeeze(0).cpu().numpy()
                    else:
                        raise ValueError(f"Model {self.embedding_model_name} is not supported")
                    embeddings.append(cls_embedding)
        else:
            with torch.no_grad():
                embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
            embeddings = list(embeddings)
        return embeddings

    def _process_split_data(self, note_data: List[List[str]], labels: List[int], split_name: str) -> tuple:
        """Process one split of data (train, validation, or test)."""
        logger.info(f"Processing {split_name} split with {len(note_data)} notes...")
        if not isinstance(note_data, list):
            raise ValueError(f"note_data for {split_name} must be List[List[str]], got {type(note_data)}, note_data: {note_data}")
        note_embeddings = []
        valid_labels = []
        original_sentences = []
        assert len(note_data) == len(labels), f"Length of note_data and labels must be the same, got {len(note_data)} and {len(labels)}"
        for i, note in tqdm(enumerate(note_data), total=len(note_data), desc=f"Processing {split_name} split"):
            
            if not isinstance(note, list):
                raise ValueError(f"Each note in {split_name} data must be a list of sentences, got {type(note)} at index {i}")

            sentences = []
            indexed_sentences = []
            for j, text in enumerate(note):
                sentences_each_note = str(text).replace('..', '').split('.')
                # Filter out empty sentences and split long sentences into segments
                filtered_sentences = []
                for s in sentences_each_note:
                    if len(s) > 128:
                        # Split into segments of 128 words or less
                        for k in range(0, len(s), 128):
                            segment = s[k:k+128]
                            filtered_sentences.append(segment.strip())
                    else:
                        filtered_sentences.append(s.strip())
                sentences_each_note = filtered_sentences[-32:]
                sentences_each_note = [sentence.strip() for sentence in sentences_each_note if (sentence.strip() and (len(sentence.strip()) > 5))]
                
                # Track source metadata for each sentence
                for sentence in sentences_each_note:
                    sentences.append(sentence)
                    indexed_sentences.append({
                        'text': sentence,
                        'note_idx': i,
                        'source_paragraph': j,
                        'sentence_idx': len(sentences) - 1
                    })
          
            if sentences:  # Only process notes with valid sentences
                embeddings = self._embed_sentences(sentences)
                if embeddings:  # Only add if we got valid embeddings
                    note_embeddings.append(embeddings)
                    valid_labels.append(labels[i])
                    original_sentences.append(indexed_sentences)  # Save structured sentences with metadata
            
        return note_embeddings, np.array(valid_labels), original_sentences
    

    def _load_pickle(self, path: str):
        """Load object from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_pickle(self, obj, path: str):
        """Save object to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def _extract_features(self, train_data: List[List[str]], train_labels: List[int],
                         validation_data: List[List[str]], validation_labels: List[int],
                         test_data: List[List[str]], test_labels: List[int], year: int, task_type: str) -> Dict:
        """Extract features for all splits and return file paths."""

        # Generate unique hex ID for this feature extraction run
        
        # Save paths for features and labels for each split
        os.makedirs(os.path.join(self.features_output_dir, self.embedding_model_name), exist_ok=True)
        train_feat_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"train_features_{year}yr_{task_type}.pkl")   
        train_label_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"train_labels_{year}yr_{task_type}.npy")
        train_sentences_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"train_sentences_{year}yr_{task_type}.pkl")
        val_feat_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"validation_features_{year}yr_{task_type}.pkl")
        val_label_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"validation_labels_{year}yr_{task_type}.npy")
        val_sentences_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"validation_sentences_{year}yr_{task_type}.pkl")
        extraction_id = uuid.uuid4().hex[:8]
        test_feat_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"test_features_{extraction_id}_{year}yr_{task_type}.pkl")  
        test_label_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"test_labels_{extraction_id}_{year}yr_{task_type}.npy")
        test_sentences_path = os.path.join(self.features_output_dir, self.embedding_model_name, f"test_sentences_{extraction_id}_{year}yr_{task_type}.pkl")
        self.train_feat_path = train_feat_path
        self.train_label_path = train_label_path
        self.train_sentences_path = train_sentences_path

        if not os.path.exists(train_feat_path) or not os.path.exists(train_label_path):
            logger.info("Processing train data...")
            train_embeddings, train_labels_array, train_original_sentences = self._process_split_data(
                train_data, train_labels, "train"
            )
            
            self._save_pickle(train_embeddings, train_feat_path)
            self._save_pickle(train_original_sentences, train_sentences_path)
            np.save(train_label_path, train_labels_array)

        if not os.path.exists(val_feat_path) or not os.path.exists(val_label_path):
            logger.info("Processing validation data...")
            val_embeddings, val_labels_array, val_original_sentences = self._process_split_data(
                validation_data, validation_labels, "validation"
            )
            self._save_pickle(val_embeddings, val_feat_path)
            self._save_pickle(val_original_sentences, val_sentences_path)
            np.save(val_label_path, val_labels_array)

        logger.info("Processing test data...")
        test_embeddings, test_labels_array, test_original_sentences = self._process_split_data(
            test_data, test_labels, "test"
        )
        self._save_pickle(test_embeddings, test_feat_path)
        self._save_pickle(test_original_sentences, test_sentences_path)
        np.save(test_label_path, test_labels_array)

        return {
            "train_features_path": train_feat_path,
            "train_sentences_path": train_sentences_path,
            "train_labels_path": train_label_path,
            "validation_features_path": val_feat_path,
            "validation_sentences_path": val_sentences_path,
            "validation_labels_path": val_label_path,
            "test_features_path": test_feat_path,
            "test_sentences_path": test_sentences_path,
            "test_labels_path": test_label_path
        }


    def _train_model(self, train_features_path: str, train_labels_path: str,
                    val_features_path: str, val_labels_path: str,
                    epochs: int, batch_size: int, learning_rate: float,
                    max_len: int, save_name: str, embedding_dim: int, year: int) -> Dict:
        """Train the model using extracted features, with learning rate scheduler."""
        if year == 0:
            save_name = f"{save_name}_classification_diagnosis_{year}yr"
        else:
            save_name = f"{save_name}_classification_prediction_{year}yr"
        
        if os.path.exists(os.path.join(self.models_output_dir, f"{save_name}_config.pkl")) and os.path.exists(os.path.join(self.models_output_dir, f"{save_name}.pt")):
            logger.info(f"Loading existing model from {os.path.join(self.models_output_dir, f'{save_name}_config.pkl')}")
            with open(os.path.join(self.models_output_dir, f"{save_name}_config.pkl"), "rb") as f:
                result_dict = pickle.load(f)
            return result_dict
        
        else:
            # Create datasets
            train_ds = NoteDataset(train_features_path, train_labels_path, max_len=max_len)
            val_ds = NoteDataset(val_features_path, val_labels_path, max_len=max_len)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)
    
            # Initialize model
            model = SentenceAttentionBERT(
                sentence_embed_dim=embedding_dim,
                weight_dim=embedding_dim,
                dropout=0,
                classifier_dropout=0.2
            ).to(self.device)
            model.train()

            # Training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            # Add a learning rate scheduler that reduces LR if validation accuracy plateaus
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
            )

            best_auc = 0.0
            train_loss = []
            val_loss = []
            val_auc = []  # Track AUC instead of accuracy
            val_auprc = []
            best_state = None

            logger.info(f"Starting training for {epochs} epochs...")

            for epoch in range(epochs):
                model.train()
                total_loss = 0
                yb_train = []   
                probs_train = []
                for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits, _, _ = model(xb)
                    loss = criterion(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    total_loss += loss.item()
                    train_loss.append(loss.item())
                    yb_train.extend(yb.cpu().numpy().flatten())
                    probs_train.extend(torch.sigmoid(logits.detach()).cpu().numpy().flatten())
                train_auc = roc_auc_score(np.array(yb_train), np.array(probs_train))
                # Validation - Calculate AUC-ROC for imbalanced data
                model.eval()
                val_epoch_loss = 0
                all_probs = []
                all_labels = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        logits, _, _ = model(xb)
                        batch_loss = criterion(logits, yb).item()
                        val_loss.append(batch_loss)
                        val_epoch_loss += batch_loss * xb.size(0)
                        
                        # Collect probabilities for AUC calculation
                        probs = torch.sigmoid(logits)
                        all_probs.extend(probs.cpu().numpy().flatten())
                        all_labels.extend(yb.cpu().numpy().flatten())
                # Calculate AUC-ROC as primary metric for imbalanced data
                auc_roc = roc_auc_score(all_labels, all_probs)
                auprc = average_precision_score(all_labels, all_probs)
                val_auc.append(auc_roc)
                val_auprc.append(auprc)
                
                logger.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train AUC-ROC: {train_auc:.4f}, Val AUC-ROC: {auc_roc:.4f}, Val AUPRC: {auprc:.4f}")
                
                # Use AUC-ROC for model selection and scheduling
                scheduler.step(auc_roc)
                save_path = os.path.join(self.models_output_dir, f"{save_name}.pt")
                if auc_roc > best_auc:
                    best_auc = auc_roc
                    best_state = model.state_dict()
                    # Save best model
                    torch.save(best_state, save_path)

            
            result_dict = {
                "model_path": save_path,
                "model_performance": {
                    "train_loss_history": train_loss,
                    "val_loss_history": val_loss,
                    "val_auc_history": val_auc,
                    "val_auprc_history": val_auprc,
                    "best_val_auc": best_auc,
                },
                "training_hyperparameters": {
                    "model_type": "SentenceAttentionBERT, a transformer-based model that uses attention to classify notes",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "embedding_model_name": self.embedding_model_name,
                    "max_len": max_len,
                    "embedding_dim": embedding_dim
                },
                }
            
            # Also save result_dict for reproducibility and metadata tracking
            result_dict_path = os.path.join(self.models_output_dir, f"{save_name}_config.pkl")
            with open(result_dict_path, "wb") as f:
                pickle.dump(result_dict, f)

            return result_dict

    def _train_cox_model(self, train_features_path: str, train_labels_path: str,
                    val_features_path: str, val_labels_path: str,
                    epochs: int, batch_size: int, learning_rate: float,
                    max_len: int, save_name: str, embedding_dim: int, year: int) -> Dict:  
        """Train Cox model using pycox, with learning rate scheduler."""
        
        save_name = f"{save_name}_survival_{year}yr_cox"
        model_path = os.path.join(self.models_output_dir, f"{save_name}.pt")
        config_path = os.path.join(self.models_output_dir, f"{save_name}_config.pkl")

        if os.path.exists(config_path) and os.path.exists(model_path):
            logger.info(f"Loading existing Cox model from {config_path}")
            with open(config_path, "rb") as f:
                result_dict = pickle.load(f)
            return result_dict
        
        # Create datasets
        train_ds = NoteCoxDataset(train_features_path, train_labels_path, max_len=max_len)
        val_ds = NoteCoxDataset(val_features_path, val_labels_path, max_len=max_len)
        np.load(val_labels_path, allow_pickle=True)
        
        logger.info(f"Training data: {len(train_ds)} samples, {train_ds.events.sum():.0f} events ({100*train_ds.events.mean():.2f}%)")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Initialize model - use the same architecture as classification
        model = SentenceAttentionBERT(
            sentence_embed_dim=embedding_dim,
            weight_dim=embedding_dim,
            dropout=0.0,
            classifier_dropout=0.2  # Add dropout to prevent overfitting
        ).to(self.device)
        model.train()

        # Training loop with Cox loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-4)
        criterion = CoxPHLoss()

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )

        best_c_index = 0.0
        train_loss = []
        train_c_index_history = []
        val_loss = []
        val_c_index_history = []

        best_state = None

        logger.info(f"Starting Cox model training for {epochs} epochs...")

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for xb, time_b, event_b in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                xb, time_b, event_b = xb.to(self.device), time_b.to(self.device), event_b.to(self.device)

                # CRITICAL: Sort by descending time for correct risk sets in Cox loss
                sorted_idx = torch.argsort(time_b, descending=True)
                xb = xb[sorted_idx]
                time_b = time_b[sorted_idx]
                event_b = event_b[sorted_idx]

                logits, _, _ = model(xb)
                # logits already has correct shape [batch_size] from model

                # Skip batches with no events (all censored) as they cause NaN in Cox loss
                if event_b.sum() == 0:
                    logger.debug(f"Batch with all censored samples detected, skipping")
                    continue

                # CoxPH loss expects (log_hazard, duration, event)
                loss = criterion(logits, time_b, event_b)
                
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf loss detected at epoch {epoch+1}, skipping batch")
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                train_loss.append(loss.item())

            # Calculate training C-index
            model.eval()
            train_risks = []
            train_times = []
            train_events = []
            
            with torch.no_grad():
                for xb, time_b, event_b in train_loader:
                    xb, time_b, event_b = xb.to(self.device), time_b.to(self.device), event_b.to(self.device)
                    logits, _, _ = model(xb)
                    # logits already has correct shape [batch_size] from model
                    
                    # Clip logits to prevent overflow in exp
                    logits = torch.clamp(logits, min=-20, max=20)
                    
                    # Convert log-hazard to cumulative hazard (proper risk score)
                    # hazard = exp(log_hazard), then cumulative hazard = -log(prod(1-h_i))
                    # For single output Cox model, we use exp(logit) as relative risk
                    hazards = torch.exp(logits)
                    train_risks.extend(hazards.cpu().numpy().flatten())
                    train_times.extend(time_b.cpu().numpy().flatten())
                    train_events.extend(event_b.cpu().numpy().flatten())
            
            # Calculate train C-index
            train_events_bool = np.array(train_events, dtype=bool)
            train_times_arr = np.array(train_times)
            train_risks_arr = np.array(train_risks)
            
            # Check for NaN or Inf values and handle them
            if np.any(np.isnan(train_risks_arr)) or np.any(np.isinf(train_risks_arr)):
                logger.warning(f"NaN or Inf detected in training risks at epoch {epoch+1}, skipping C-index calculation")
                train_c_index = 0.5  # Default value
            else:
                # Hazard/relative risk is already proper (higher = more risk), no negation needed
                train_c_result = concordance_index_censored(train_events_bool, train_times_arr, train_risks_arr)
                train_c_index = train_c_result[0]
            train_c_index_history.append(train_c_index)

            # Validation - Calculate C-index
            model.eval()
            val_epoch_loss = 0
            all_risks = []
            all_times = []
            all_events = []
            
            with torch.no_grad():
                for xb, time_b, event_b in val_loader:
                    xb, time_b, event_b = xb.to(self.device), time_b.to(self.device), event_b.to(self.device)
     
                    # Sort validation batch for consistent loss computation
                    sorted_idx = torch.argsort(time_b, descending=True)
                    xb = xb[sorted_idx]
                    time_b = time_b[sorted_idx]
                    event_b = event_b[sorted_idx]

                    logits, _, _ = model(xb)
                    # logits already has correct shape [batch_size] from model
                    
                    batch_loss = criterion(logits, time_b, event_b).item()
                    val_loss.append(batch_loss)
                    val_epoch_loss += batch_loss * xb.size(0)
                    
                    # Clip logits to prevent overflow in exp
                    logits = torch.clamp(logits, min=-20, max=20)
                    
                    # Convert log-hazard to hazard/relative risk (proper risk score)
                    hazards = torch.exp(logits)
                    all_risks.extend(hazards.cpu().numpy().flatten())
                    all_times.extend(time_b.cpu().numpy().flatten())
                    all_events.extend(event_b.cpu().numpy().flatten())
            
            # Calculate C-index
            all_events_bool = np.array(all_events, dtype=bool)
            all_times = np.array(all_times)
            all_risks = np.array(all_risks)
            
            # Check for NaN or Inf values and handle them
            if np.any(np.isnan(all_risks)) or np.any(np.isinf(all_risks)):
                logger.warning(f"NaN or Inf detected in validation risks at epoch {epoch+1}, skipping C-index calculation")
                c_index = 0.5  # Default value
            else:
                # Hazard/relative risk is already proper (higher = more risk), no negation needed
                c_result = concordance_index_censored(all_events_bool, all_times, all_risks)
                c_index = c_result[0]
            val_c_index_history.append(c_index)
            
            logger.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train C-index: {train_c_index:.4f}, "
                       f"Val C-index: {c_index:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, "
                       f"Train risks: [{np.min(train_risks_arr):.3f}, {np.max(train_risks_arr):.3f}], "
                       f"Val risks: [{np.min(all_risks):.3f}, {np.max(all_risks):.3f}]")
            
            # Use C-index for model selection and scheduling
            scheduler.step(c_index)
            save_path = os.path.join(self.models_output_dir, f"{save_name}.pt")
            if c_index > best_c_index:
                best_c_index = c_index
                best_state = model.state_dict()
                torch.save(best_state, save_path)
        # Save best model (or final model if no improvement)
        
        if best_state is None:
            logger.warning("No best model found (all C-indices were <= 0), saving final model state")
            best_state = model.state_dict()
            torch.save(best_state, save_path)
        
        result_dict = {
            "model_path": save_path,
            "model_performance": {
                "train_loss_history": train_loss,
                "train_c_index_history": train_c_index_history,
                "val_loss_history": val_loss,
                "val_c_index_history": val_c_index_history,
                "best_val_c_index": best_c_index
            },
            "training_hyperparameters": {
                "model_type": "SentenceAttentionBERT with CoxPH loss for survival analysis",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "embedding_model_name": self.embedding_model_name,
                "max_len": max_len,
                "embedding_dim": embedding_dim
            },
        }
        
        torch.save(best_state, save_path)
        result_dict_path = os.path.join(self.models_output_dir, f"{save_name}_config.pkl")
        with open(result_dict_path, "wb") as f:
            pickle.dump(result_dict, f)

        return result_dict


    def execute(self,
                train_data_path: str,
                train_labels_path: str,
                validation_data_path: str,
                validation_labels_path: str,
                test_data_path: str,
                test_labels_path: str,
                embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                epochs: int = 40,
                batch_size: int = 256,
                learning_rate: float = 1e-4,
                max_len: int = 128,
                save_name: str = "note_trained_model",
                year: int = 1,
                task_type: str = "classification",
                **kwargs):
        """
        Execute the complete trainer: load data from path → extract features → train model.

        Args:
            *_path: Paths to saved .pkl (data) and .npy (labels) files.
            embedding_model_name: Name of the embedding model to use.
            task_type: Type of task ('classification' or 'survival').
            Other training hyperparameters...

        Returns:
            Metadata object with saved model path and extracted feature paths.
        """

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_model_name = embedding_model_name
            logger.info(f"Initializing Note_Trainer_Tool with model: {embedding_model_name}, device: {self.device}")

            # Load data
            with open(train_data_path, 'rb') as f:
                train_data = pickle.load(f)
            train_labels = np.load(train_labels_path, allow_pickle=True)

            

            with open(validation_data_path, 'rb') as f:
                validation_data = pickle.load(f)
            validation_labels = np.load(validation_labels_path, allow_pickle=True)

            with open(test_data_path, 'rb') as f:
                test_data = pickle.load(f)
            test_labels = np.load(test_labels_path, allow_pickle=True)

 
            # Initialize embedding model
  
            if embedding_model_name in ['roberta-base', 'Qwen/Qwen2.5-0.5B']:
                self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
                self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
                self.embedding_model.eval()
                self.embedding_dim = 768 if embedding_model_name == "roberta-base" else 1024
            elif embedding_model_name in ['Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-8B']:
                from sentence_transformers import SentenceTransformer
                from transformers import logging
                logging.set_verbosity_error()
                self.embedding_model = SentenceTransformer(embedding_model_name)
                if embedding_model_name == 'Qwen/Qwen3-Embedding-8B':
                    self.embedding_dim = 4096
                elif embedding_model_name == 'Qwen/Qwen3-Embedding-4B':
                    self.embedding_dim = 2560
                elif embedding_model_name == 'Qwen/Qwen3-Embedding-0.6B':
                    self.embedding_dim = 1024
                self.embedding_model.eval()
            else:
                raise ValueError(f"Model {embedding_model_name} is not supported")

            # === Feature extraction ===
            logger.info("Starting feature extraction phase...")
            feature_paths = self._extract_features(
                train_data, train_labels,
                validation_data, validation_labels,
                test_data, test_labels, year, task_type
            )
            logger.info("Feature extraction completed.")

            # === Model training ===
            if task_type == "survival":
                logger.info("Training Cox survival model...")
                training_results = self._train_cox_model(
                    feature_paths["train_features_path"],
                    feature_paths["train_labels_path"],
                    feature_paths["validation_features_path"],
                    feature_paths["validation_labels_path"],
                    epochs,
                    batch_size,
                    learning_rate,
                    max_len,
                    save_name,
                    embedding_dim=self.embedding_dim,
                    year=year,
                )
                logger.info(f"Cox model training complete. Saved to: {training_results['model_path']}")
                metric_name = "best_val_c_index"
                metric_value = training_results['model_performance'][metric_name]
            elif "classification" in task_type:
                logger.info("Training classification model...")
                training_results = self._train_model(
                    feature_paths["train_features_path"],
                    feature_paths["train_labels_path"],
                    feature_paths["validation_features_path"],
                    feature_paths["validation_labels_path"],
                    epochs,
                    batch_size,
                    learning_rate,
                    max_len,
                    save_name,
                    embedding_dim=self.embedding_dim,
                    year=year,
                )
                logger.info(f"Model training complete. Saved to: {training_results['model_path']}")
                metric_name = "best_val_auc"
                metric_value = training_results['model_performance'][metric_name]
            else:
                raise ValueError(f"Invalid task type: {task_type}")

            # Create metadata using the new structure
            return Metadata.create_agent_output(
                status="success",
                dataset={
                    "train_features": {
                        "saved_path": feature_paths["train_features_path"],
                        "description": "Train features extracted from notes",
                        "configuration": {"dataset_len": len(train_data)}
                    },
                    "validation_features": {
                        "saved_path": feature_paths["validation_features_path"],
                        "description": "Validation features extracted from notes",
                        "configuration": {"dataset_len": len(validation_data)}
                    },
                    "test_features": {
                        "saved_path": feature_paths["test_features_path"],
                        "description": "Test features extracted from notes",
                        "configuration": {"dataset_len": len(test_data)}
                    },
                    "train_sentences": {
                        "saved_path": feature_paths["train_sentences_path"],
                        "description": "sentences by splitting original training notes",
                        "configuration": {"dataset_len": len(train_data)}
                    },
                    "validation_sentences": {
                        "saved_path": feature_paths["validation_sentences_path"],
                        "description": "sentences by splitting original validation notes",
                        "configuration": {"dataset_len": len(validation_data)}
                    },
                    "test_sentences": {
                        "saved_path": feature_paths["test_sentences_path"],
                        "description": "sentences by splitting original test notes",
                        "configuration": {"dataset_len": len(test_data)}
                    },
                    'model_performance': {
                        "saved_path": None,
                        "description": f"Validation {metric_name} for the trained model",
                        "configuration": {"model_performance": metric_value}
                    }
                },
                model={
                    "trained_model": {
                        "saved_path": training_results["model_path"],
                        "description": f"{'Binary classification' if 'classification' in task_type else 'Cox survival analysis'} model trained on note embeddings",
                        "configuration": {
                            metric_name: metric_value,
                            "training_hyperparameters": training_results["training_hyperparameters"]
                        }
                    }
                },
                cache_directory=CACHE_DIR,
                agent_name="note_agent"
            )

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                filename, lineno, func, text = tb[-1]
                logger.error(
                    f"Error during pipeline execution: {str(e)}\n"
                    f"Occurred in file: {filename}, line {lineno}, in {func}\n"
                    f"Code: {text}"
                )
            else:
                logger.error(f"Error during pipeline execution: {str(e)}")

            return Metadata.create_agent_output(
                status="error",
                dataset={
                    "error": {
                        "saved_path": None,
                        "description": str(e),
                        "configuration": {
                            "train_data_path": train_data_path,
                            "validation_data_path": validation_data_path,
                            "test_data_path": test_data_path,
                            "embedding_model_name": embedding_model_name,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "max_len": max_len
                        }
                    }
                },
                cache_directory=CACHE_DIR,
                agent_name="note_agent"
            )



if __name__ == "__main__":
    # Test the note trainer tool
    from cerebra.agents.data_agent import DataAgent
    year = 1
    time_to_event = True
    # year = 1
    # time_to_event = False
    data_agent = DataAgent()
    input_metadata = data_agent.run(f"Load initial data for note_agent", agent_name='note_agent', patient_id=66, year=year, time_to_event=time_to_event)
    tool = Note_Model_Trainer_Tool()
    result = tool.execute(
        train_data_path=input_metadata.get_metadata_info()['dataset']['train_data']['saved_path'],
        train_labels_path=input_metadata.get_metadata_info()['dataset']['train_labels']['saved_path'],
        validation_data_path=input_metadata.get_metadata_info()['dataset']['validation_data']['saved_path'],
        validation_labels_path=input_metadata.get_metadata_info()['dataset']['validation_labels']['saved_path'],
        test_data_path=input_metadata.get_metadata_info()['dataset']['test_data']['saved_path'],
        test_labels_path=input_metadata.get_metadata_info()['dataset']['test_labels']['saved_path'],
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        epochs=6,  # Small number for testing
        batch_size=256, 
        learning_rate=5e-4,
        max_len=256,
        task_type="survival" if time_to_event else "classification",
        save_name="note_trained_model_longer_training" if time_to_event else "note_trained_model_classification",
        year=year
    )
 
    # Extract data from Dataset result
    metadata_dict = result.get_metadata_info()

    if metadata_dict["status"] == "success":
        print("✅ Pipeline completed successfully!")
        print(f"Model saved to: {metadata_dict['model']['trained_model']['saved_path']}")
        print("\nFeature files:")
        for name, info in metadata_dict['dataset'].items():
            print(f"  {name}: {info['saved_path']}")

        print(f"\nTraining metadata: {metadata_dict['model']['trained_model']['configuration']}")
    else:
        print(f"❌ Pipeline failed: see logs for details.")