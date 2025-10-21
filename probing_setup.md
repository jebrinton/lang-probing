# Lang Probing System

System for training linear probes on MLP layer activations and analyzing features correlated with grammatical concepts.

## Project Structure

```
lang-probing/
├── src/                   # Source code
│   ├── config.py          # Global configuration
│   ├── data.py            # Data loading from UD treebanks
│   ├── activations.py     # MLP activation extraction
│   ├── probe.py           # Probe training
│   ├── features.py        # Correlated feature analysis
│   ├── ablation.py        # Ablation experiments
│   └── utils.py           # General utilities
├── tests/                 # Unit tests
├── scripts/               # Execution scripts
│   ├── verify_setup.py    # Verify setup
│   ├── train_probes.py    # Train probes
│   ├── find_features.py   # Find top features
│   └── run_ablation.py    # Run ablation experiments
├── outputs/               # Results
│   ├── probes/            # Trained probes (.joblib)
│   ├── features/          # Top features (.json)
│   └── ablations/         # Ablation results (.json)
└── logs/                  # Execution logs
```

## Initial Setup

### 1. Verify the Setup

```bash
cd /projectnb/mcnet/jbrin/lang-probing
python scripts/verify_setup.py
```

This script checks:

* Required directories
* Availability of UD treebanks
* Python dependencies
* GPU access
* Access to the model and SAE
* Data loading

### 2. Configuration

Edit `src/config.py` to adjust:

* `LANGUAGES`: Languages to process (e.g.: English, Spanish, Turkish)
* `CONCEPTS`: Grammatical concepts to analyze
* `BATCH_SIZE`, `SEED`, etc.

## Usage Pipeline

### Step 1: Train Probes

Train linear probes for all concepts in all languages:

```bash
python scripts/train_probes.py
```

Options:

* `--languages LANG1 LANG2`: Process specific languages
* `--concepts CONCEPT1 CONCEPT2`: Process specific concepts
* `--min_samples N`: Minimum required samples (default: 128)
* `--overwrite`: Overwrite existing probes

Example:

```bash
python scripts/train_probes.py --languages English --concepts Tense Number
```

Probes are saved in `outputs/probes/` with the format:
`{language}_{concept_key}_{concept_value}.joblib`

### Step 2: Find Top Features

Analyze the trained probes to identify the most correlated SAE features:

```bash
python scripts/find_features.py
```

Options:

* `--k N`: Number of top features (default: 100)
* `--concept_key KEY --concept_value VALUE`: Process a specific concept

Example:

```bash
python scripts/find_features.py --k 50 --concept_key Tense --concept_value Past
```

Results are saved in `outputs/features/`:

* `{concept_key}_{concept_value}.json`: Features per language
* `{concept_key}_{concept_value}_shared.json`: Features shared across languages

### Step 3: Ablation Experiments

Ablate specific features and measure the effect on classification:

```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Tense \
    --concept_value Past \
    --examples examples/tense_past_english.txt \
    --k 10 \
    --mode simple
```

Options:

* `--language LANG`: Language to test
* `--concept_key KEY`: Concept key
* `--concept_value VALUE`: Concept value
* `--examples FILE`: Example file (see format below)
* `--k N`: Number of top features to ablate
* `--mode MODE`: Ablation mode:

  * `simple`: Simple ablation on all examples
  * `progressive`: Progressive ablation (adds features one by one)
  * `necessity`: Feature necessity test

#### Example File Format

```
positive
The boy walked to school yesterday.
She ate breakfast this morning.

negative
The boy walks to school every day.
She eats breakfast every morning.
```

## Running Tests

```bash
cd /project/directory

# Run all tests
python -m pytest tests/

# Run specific tests
python -m pytest tests/test_data.py
python -m pytest tests/test_probe.py
python -m pytest tests/test_features.py

# Run with unittest
python tests/test_data.py
```

## Data Structure

### Probes

Probes are `LogisticRegression` models from sklearn saved with joblib.

### Features

JSON file with structure:

```json
{
  "English": [
    [12345, 0.8542],  // [feature_index, weight]
    [67890, 0.7321],
    ...
  ],
  "Spanish": [
    [12345, 0.7892],
    [11223, 0.6543],
    ...
  ]
}
```

### Shared Features

```json
{
  "12345": {
    "count": 3,
    "languages": ["English", "Spanish", "Turkish"],
    "avg_weight": 0.8123,
    "std_weight": 0.0432
  }
}
```

### Ablation Results

```json
{
  "language": "English",
  "concept_key": "Tense",
  "concept_value": "Past",
  "k": 10,
  "experiments": [
    {
      "sentence": "The boy walked to school.",
      "label": "positive",
      "original_logit": 2.345,
      "ablated_logit": -0.123,
      "logit_change": -2.468,
      "original_prob": 0.912,
      "ablated_prob": 0.469,
      "prob_change": -0.443
    }
  ]
}
```

## Default Languages and Concepts

### Languages

* English (multi-treebank: EWT, GUM, PUD, etc.)
* Spanish (multi-treebank: AnCora, GSD, PUD, etc.)
* Turkish (multi-treebank: BOUN, GB, IMST, PUD, etc.)

### Concepts

* Tense (e.g., Past, Pres, Fut)
* Number (e.g., Sing, Plur)
* Gender (e.g., Masc, Fem, Neut)

## Technical Notes

* **Model**: Llama-3.1-8B
* **SAE**: jbringma/sae-llama-3-8b-layer16 (32768 features)
* **MLP Activations**: Extracted from `model.model.layers[layer_num].output[0]`
* **Pooling**: Mean pooling over token activations (weighted by attention mask)
* **Probe**: LogisticRegression with `class_weight="balanced"`, `C=0.1`
* **Correlation**: Based on absolute probe weights
* **Paths**: Configured in `src/config.py`

## Future Extensions

To implement attribution patching (causal feature analysis):

1. Create `src/attribution.py` adapting code from `multilingual-features`
2. Add a function in `features.py` for gradient-based attribution
3. Compare results with correlation based on probe weights

## Troubleshooting

### Error: CUDA out of memory

Reduce `BATCH_SIZE` in `src/config.py`.

### Error: Not enough samples

Some concepts may not have enough examples. The system requires at least 128 balanced examples by default (adjustable with `--min_samples`).

---
