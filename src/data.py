"""
Dataset loading y procesamiento desde Universal Dependencies treebanks
"""

import os
import glob
import numpy as np
import pyconll
from torch.utils.data import Dataset
from sklearn.utils import resample
import logging



def get_all_treebank_files(language, split, ud_base_folder=None):
    """
    Obtiene todos los archivos .conllu para un idioma y split específicos.
    
    Args:
        language: Nombre simple del idioma (e.g., 'English')
        split: 'train', 'dev', 'test'
        ud_base_folder: Path base de UD (usa config si es None)
        
    Returns:
        list: Lista de paths a archivos .conllu
    """
    if ud_base_folder is None:
        from .config import UD_BASE_FOLDER
        ud_base_folder = UD_BASE_FOLDER
    
    # Buscar todos los directorios que empiecen con UD_{language}-
    pattern = os.path.join(ud_base_folder, f"UD_{language}-*")
    treebank_dirs = glob.glob(pattern)
    
    files = []
    for treebank_dir in treebank_dirs:
        if os.path.isdir(treebank_dir):
            # Buscar archivos que terminen en -ud-{split}.conllu
            file_pattern = os.path.join(treebank_dir, f"*-ud-{split}.conllu")
            treebank_files = glob.glob(file_pattern)
            files.extend(treebank_files)
    
    return files


def get_training_files(language, ud_base_folder=None):
    """
    Obtiene todos los archivos de entrenamiento (train + dev) para un idioma.
    
    Args:
        language: Nombre simple del idioma (e.g., 'English')
        ud_base_folder: Path base de UD (usa config si es None)
        
    Returns:
        list: Lista de paths a archivos .conllu para entrenamiento
    """
    train_files = get_all_treebank_files(language, 'train', ud_base_folder)
    dev_files = get_all_treebank_files(language, 'dev', ud_base_folder)
    return train_files + dev_files


def get_test_files(language, ud_base_folder=None):
    """
    Obtiene todos los archivos de test para un idioma.
    
    Args:
        language: Nombre simple del idioma (e.g., 'English')
        ud_base_folder: Path base de UD (usa config si es None)
        
    Returns:
        list: Lista de paths a archivos .conllu para test
    """
    return get_all_treebank_files(language, 'test', ud_base_folder)


def get_ud_filepath(language, split='train', ud_base_folder=None):
    """
    DEPRECATED: Busca el primer archivo .conllu disponible para un idioma y split.
    Use get_all_treebank_files() para obtener todos los treebanks de un idioma.
    
    Args:
        language: Nombre del idioma (e.g., 'English-PUD' o 'English')
        split: 'train', 'test', o 'dev'
        ud_base_folder: Path base de UD (usa config si es None)
        
    Returns:
        str: Path al archivo .conllu, o None si no se encuentra
    """
    if ud_base_folder is None:
        from .config import UD_BASE_FOLDER
        ud_base_folder = UD_BASE_FOLDER
    
    # Intentar primero con el formato nuevo (nombre simple)
    files = get_all_treebank_files(language, split, ud_base_folder)
    if files:
        return files[0]
    
    # Fallback al formato antiguo para compatibilidad
    ud_folder = os.path.join(ud_base_folder, f"UD_{language}")
    
    if not os.path.exists(ud_folder):
        return None
    
    # Buscar archivo con el split especificado
    pattern = os.path.join(ud_folder, f"*-ud-{split}.conllu")
    files = glob.glob(pattern)
    
    return files[0] if files else None


def get_available_concepts(conll_files):
    """
    Extrae todos los conceptos gramaticales disponibles en uno o más archivos .conllu.
    
    Args:
        conll_files: Path al archivo .conllu o lista de paths
        
    Returns:
        dict: {concept_key: set(concept_values)}
    """
    # Convertir a lista si es un solo archivo
    if isinstance(conll_files, str):
        conll_files = [conll_files]
    
    features = {}
    
    for conll_file in conll_files:
        if not os.path.exists(conll_file):
            logging.warning(f"File not found: {conll_file}")
            continue
            
        data = pyconll.load_from_file(conll_file)
        
        for sentence in data:
            for token in sentence:
                if token.feats:
                    for feat, values in token.feats.items():
                        if feat not in features:
                            features[feat] = set()
                        features[feat].update(values)
    
    return features


def concept_filter(sentence, concept_key, concept_value):
    """
    Determina si una sentence contiene un concepto específico.
    
    Args:
        sentence: pyconll.Sentence object
        concept_key: Clave del concepto (e.g., 'Tense')
        concept_value: Valor del concepto (e.g., 'Past')
        
    Returns:
        bool: True si algún token tiene el concepto
    """
    for token in sentence:
        if token.feats and concept_key in token.feats:
            if concept_value in token.feats.get(concept_key, {}):
                return True
    return False


class ProbingDataset(Dataset):
    """
    Dataset para probing de conceptos gramaticales.
    
    Carga sentences desde uno o más archivos .conllu y las etiqueta basándose en
    la presencia de un concepto gramatical específico.
    """
    
    def __init__(self, conll_files, concept_key, concept_value):
        """
        Args:
            conll_files: Path al archivo .conllu o lista de paths
            concept_key: Clave del concepto (e.g., 'Tense')
            concept_value: Valor del concepto (e.g., 'Past')
        """
        self.concept_key = concept_key
        self.concept_value = concept_value
        self.sentences = []
        self.labels = []
        self.load_data(conll_files)
    
    def load_data(self, conll_files):
        """
        Carga datos desde uno o más archivos .conllu.
        
        Args:
            conll_files: Path al archivo .conllu o lista de paths
        """
        # Convertir a lista si es un solo archivo
        if isinstance(conll_files, str):
            conll_files = [conll_files]
        
        for conll_file in conll_files:
            if not os.path.exists(conll_file):
                logging.warning(f"File not found: {conll_file}")
                continue
                
            logging.info(f"Loading data from: {os.path.basename(conll_file)}")
            data = pyconll.load_from_file(conll_file)
            
            for sentence in data:
                # Etiquetar como 1 si contiene el concepto, 0 si no
                label = 1 if concept_filter(sentence, self.concept_key, self.concept_value) else 0
                
                self.sentences.append(sentence.text)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {'sentence': str, 'label': int}
        """
        return {
            "sentence": self.sentences[idx],
            "label": self.labels[idx]
        }


def balance_dataset(dataset, seed=42):
    """
    Balancea el dataset mediante undersampling de la clase mayoritaria.
    
    Args:
        dataset: ProbingDataset o lista de dicts con 'label'
        seed: Random seed para reproducibilidad
        
    Returns:
        list: Dataset balanceado
    """
    # Convertir a lista si es necesario
    if isinstance(dataset, ProbingDataset):
        items = [dataset[i] for i in range(len(dataset))]
    else:
        items = list(dataset)
    
    # Separar por clase
    positive_samples = [item for item in items if item['label'] == 1]
    negative_samples = [item for item in items if item['label'] == 0]
    
    logging.info(f"Before balancing:")
    logging.info(f"  Total samples: {len(items)}")
    logging.info(f"  Positive samples: {len(positive_samples)}")
    logging.info(f"  Negative samples: {len(negative_samples)}")
    
    # Verificar que hay suficientes muestras
    if not positive_samples or not negative_samples:
        logging.warning("No positive or negative samples found.")
        return None
    
    # Undersample la clase mayoritaria
    if len(positive_samples) < len(negative_samples):
        negative_samples = resample(
            negative_samples, 
            n_samples=len(positive_samples), 
            random_state=seed
        )
    else:
        positive_samples = resample(
            positive_samples, 
            n_samples=len(negative_samples), 
            random_state=seed
        )
    
    # Combinar y mezclar
    balanced_dataset = positive_samples + negative_samples
    np.random.seed(seed)
    np.random.shuffle(balanced_dataset)
    
    logging.info(f"After balancing:")
    logging.info(f"  Total samples: {len(balanced_dataset)}")
    logging.info(f"  Positive samples: {len(positive_samples)}")
    logging.info(f"  Negative samples: {len(negative_samples)}")
    
    return balanced_dataset

