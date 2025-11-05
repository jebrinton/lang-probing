"""
Configuración global del sistema de probing
"""
import os

# Base directory for the project
BASE_DIR = "/projectnb/mcnet/jbrin/lang-probing"

# Output directories
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PROBES_DIR = os.path.join(OUTPUTS_DIR, "probes")
FEATURES_DIR = os.path.join(OUTPUTS_DIR, "features")
ABLATIONS_DIR = os.path.join(OUTPUTS_DIR, "ablations")
ACTIVATIONS_DIR = os.path.join(OUTPUTS_DIR, "activations")
STEERING_VECTORS_DIR = os.path.join(OUTPUTS_DIR, "steering_vectors/all")
STEERING_VECTORS_DIR_NOVA = os.path.join(STEERING_VECTORS_DIR, "nova")
STEERING_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "steering")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Paths
UD_BASE_FOLDER = "/projectnb/mcnet/jbrin/.cache/ud/ud-treebanks-v2.16/"

# Model configuration
MODEL_ID = "meta-llama/Llama-3.1-8B"
SAE_ID = "jbrinkma/sae-llama-3-8b-layer16"
LAYER_NUM = 16
SAE_DIM = 32768

# Languages to process (expandible)
LANGUAGES_NOVA = ["English", "Spanish", "Turkish", "Arabic", "German", "Chinese", "French", "Japanese"]
LANGUAGES = [
    "Arabic", "Chinese", "Czech", "Dutch", "English", "French", 
    "German", "Greek", "Hebrew", "Hindi", "Indonesian", "Italian", 
    "Japanese", "Korean", "Persian", "Polish", "Portuguese", "Romanian", 
    "Russian", "Spanish", "Turkish", "Ukrainian", "Vietnamese"
]

# Grammatical concepts to probe
CONCEPTS_NOVA = ["Tense", "Number"]
CONCEPTS = [
    "Gender", "Animacy", "Mood", "Tense",
    "Reflex", "Number", "Aspect", "Case",
    "Definite", "Evident", "Polarity", 
    "Person", "Degree", "Polite"
]

CONCEPTS_VALUES = {
    "Gender": ["Masc", "Fem", "Neut"],
    "Number": ["Sing", "Dual", "Plur"],
    "Tense": ["Past", "Pres", "Fut"],
    # "Animacy": ["Anim", "Inan"],
    "Case": ["Nom", "Acc", "Gen", "Dat", "Loc"],
    "Polarity": ["Pos", "Neg"],
    "Aspect": ["Prog", "Imp", "Perf"],
    "Mood": ["Ind", "Imp", "Cnd", "Sub"],
    "Polite": ["Infm", "Form"],
    "Person": ["1", "2", "3"],
    "Degree": ["Pos", "Cmp", "Sup"],
}

LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


LANGUAGES = ["English", "Spanish"]
CONCEPTS_VALUES = {
    "Gender": ["Masc", "Fem"],
}
LAYERS = [16]


# Training parameters
BATCH_SIZE = 16
SEED = 42
MAX_ITER = 5000
C_VALUE = 0.1  # Regularization parameter for LogisticRegression

# NNsight tracer configuration
TRACER_KWARGS = {'scan': False, 'validate': False}

# Feature analysis
TOP_K_FEATURES = 100  # Number of top correlating features to save

# Steering vectors configuration
MIN_SAMPLES_FOR_STEERING = 100  # Minimum samples required for steering vector generation
MAX_SAMPLES_FOR_STEERING = 8000  # Maximum sentences https://nnsight.net/notebooks/features/multiple_token/to process for steering vector generation

# Activation collection configuration
COLLECTION_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]  # Layers to extract activations from (every 4th layer)
