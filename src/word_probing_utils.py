import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import logging

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
    """Hyperparameter search for the logistic regression probe."""

    # get number of CPU
    num_cpu = int(os.environ.get("NSLOTS", 1))

    probe_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            random_state=seed,
            class_weight="balanced"
        ))
    ])

    param_grid = [{
        'model__C' : np.logspace(-4, 1, 5),
        'model__penalty': ['l2'],
        'model__solver': ['saga'],
        'model__max_iter': [1000]
    }]

    grid_search = GridSearchCV(
        probe_pipeline,
        param_grid, 
        cv=3,
        scoring='accuracy',
        n_jobs=num_cpu-1,
        verbose=2
    )
    
    grid_search.fit(train_activations, train_labels)
    best_classifier = grid_search.best_estimator_
    logging.info(f"Best classifier: {best_classifier}")
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best score: {grid_search.best_score_}")
    logging.info(f"Coefficients: {best_classifier.named_steps['model'].coef_}")
    logging.info(f"Num non-zero coeffs: {np.count_nonzero(best_classifier.named_steps['model'].coef_)}")
    return best_classifier

def train_and_evaluate_probe(train_activations, train_labels, test_activations, test_labels, seed):
    """Train a logistic regression probe and evaluate its performance."""
    logging.info("Training logistic regression model...")
    classifier = get_best_classifier(train_activations, train_labels, seed)
    classifier.fit(train_activations, train_labels)

    train_accuracy = classifier.score(train_activations, train_labels)
    test_accuracy = classifier.score(test_activations, test_labels)

    logging.info(f"Train Accuracy: {train_accuracy:.2f}")
    logging.info(f"Test Accuracy: {test_accuracy:.2f}")

    return classifier