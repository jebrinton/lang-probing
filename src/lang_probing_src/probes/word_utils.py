import os
import numpy as np
from torch.utils.data import Dataset
import logging

import cupy
from cuml.preprocessing import StandardScaler
from cuml.pipeline import Pipeline
from cuml.model_selection import GridSearchCV
from cuml.linear_model import LogisticRegression # You already have this
from sklearn.metrics import make_scorer, accuracy_score # <--- IMPORT NEW SCORERS


# This is our custom, GPU-aware scorer
def gpu_aware_accuracy(y_true, y_pred):
    """
    Custom accuracy scorer that moves y_pred from GPU (CuPy) to CPU (NumPy)
    before calling sklearn's accuracy_score.
    
    y_true will be a NumPy array (from our previous fix).
    y_pred will be a CuPy array (from the cuML model's .predict()).
    """
    if isinstance(y_pred, cupy.ndarray):
        y_pred = cupy.asnumpy(y_pred)
    
    if isinstance(y_true, cupy.ndarray):
        y_true = cupy.asnumpy(y_true)
        
    # Now both are NumPy arrays, and sklearn.metrics.accuracy_score will work
    return accuracy_score(y_true, y_pred)


class WordProbingCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch is a list of tuples: [ (word_list_1, label_list_1), (word_list_2, label_list_2), ... ]
        
        # 1. Unzip the batch
        word_form_lists = [item[0] for item in batch]
        word_label_lists = [item[1] for item in batch]

        # 2. Tokenize the word lists
        tokenized_batch = self.tokenizer(
            word_form_lists, 
            is_split_into_words=True, # important
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # 3. Return the tokenized batch and the un-stacked labels
        # The loop will get (tokenized_batch, word_label_lists)
        return tokenized_batch, word_label_lists


class WordProbingDataset(Dataset):
    def __init__(self, processed_sentences, concept_key, concept_value):
        """
        Args:
            processed_sentences (list[list[dict]]): Parsed sentences data.
            concept_key (str): The feature to probe (e.g., "Number").
            concept_value (str): The value to treat as '1' (e.g., "Plur").
        """
        self.word_forms = []
        self.word_labels = []

        for proc_sentence in processed_sentences:
            if not proc_sentence: # skip empty sentences
                continue
            
            word_forms = []
            labels = []

            for word_dict in proc_sentence:
                word_form = word_dict.get("form", "") # replace None with an empty string
                
                word_forms.append(word_form)
                
                feats = word_dict.get("feats", {})
                label = 1 if concept_value in feats.get(concept_key, set()) else 0
                labels.append(label)
            
            self.word_forms.append(word_forms)
            self.word_labels.append(labels)

    def __len__(self):
        if (len(self.word_forms) != len(self.word_labels)):
            raise ValueError("word_forms and word_labels must have the same length")
        return len(self.word_forms)

    def __getitem__(self, idx):
        return self.word_forms[idx], self.word_labels[idx]


def get_best_classifier(train_activations, train_labels, seed):
    """Hyperparameter search for the logistic regression probe (GPU-accelerated)."""

    # 1. Create the cuML Pipeline
    probe_pipeline = Pipeline([
        ('scaler', StandardScaler()), # Use cuml.preprocessing.StandardScaler
        ('model', LogisticRegression( # Use LogisticRegression
            class_weight="balanced",
            penalty='l2',
            solver='qn',
            max_iter=2000
        ))
    ])

    # 2. Update Param Grid for cuLogisticRegression
    #    cuML's LogisticRegression has different solvers.
    #    'qn' (Quasi-Newton) is a good, fast default.
    #    'l2' is the only supported penalty for 'qn'.
    param_grid = [{
        'model__C' : np.logspace(-4, 3, 16).tolist(), # np.logspace is fine, it just creates a CPU array
    }]

    # 3. Create the GPU-aware scorer
    # We use make_scorer to turn our function into a valid scorer for GridSearchCV
    custom_scorer = make_scorer(gpu_aware_accuracy)

    # 4. Use cuML's GridSearchCV, now with the custom scorer
    grid_search = GridSearchCV(
        probe_pipeline,
        param_grid,
        cv=4,
        scoring=custom_scorer,
        verbose=0
    )
    
    # 5. Convert labels to CPU (NumPy) for sklearn's CV splitter
    if isinstance(train_labels, cupy.ndarray):
        train_labels_cpu = cupy.asnumpy(train_labels)
    else:
        train_labels_cpu = train_labels
    
    # This .fit() now runs entirely on the GPU
    # It will use:
    # - X_gpu (train_activations)
    # - y_cpu (train_labels_cpu) for splitting
    # - custom_scorer for scoring (which handles y_pred_gpu)
    grid_search.fit(train_activations, train_labels_cpu)
    
    # .best_estimator_ is already fitted on the best params
    best_classifier = grid_search.best_estimator_

    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best score: {grid_search.best_score_}")
    
    if isinstance(train_labels, cupy.ndarray):
        train_labels_cpu = cupy.asnumpy(train_labels)
    else:
        train_labels_cpu = train_labels
    
    grid_search.fit(train_activations, train_labels_cpu)
    best_classifier = grid_search.best_estimator_

    return best_classifier, grid_search.best_params_, grid_search.best_score_

def train_and_evaluate_probe(train_activations, train_labels, test_activations, test_labels, seed):
    """Train a logistic regression probe and evaluate its performance."""
    logging.info("Training logistic regression model...")

    if not isinstance(train_activations, cupy.ndarray):
        train_activations = cupy.asarray(train_activations, dtype=cupy.float32)
        train_labels = cupy.asarray(train_labels, dtype=cupy.int32)
        test_activations = cupy.asarray(test_activations, dtype=cupy.float32)
        test_labels = cupy.asarray(test_labels, dtype=cupy.int32)

    classifier, best_params, cv_score = get_best_classifier(
        train_activations, train_labels, seed
    )

    train_accuracy = classifier.score(train_activations, train_labels)
    test_accuracy = classifier.score(test_activations, test_labels)

    # Convert from CuPy/Numpy types to standard Python types for serialization
    # .get() pulls from GPU, .item() converts from numpy/cupy scalar
    if hasattr(train_accuracy, 'get'):
        train_accuracy = train_accuracy.get().item()
    if hasattr(test_accuracy, 'get'):
        test_accuracy = test_accuracy.get().item()
        
    # Clean up params for serialization (e.g., np.float64 -> float)
    cleaned_params = {k: v.item() if hasattr(v, 'item') else v for k, v in best_params.items()}

    logging.info(f"Train Accuracy: {train_accuracy:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")

    stats = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_score": cv_score,
        "best_params": str(cleaned_params) # Store as a string in CSV
    }
    
    return classifier, stats