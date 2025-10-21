# Quick Start Guide

## Initial Setup (One-Time)

### 1. Verify Project Structure

```bash
cd /parent/directories/lang-probing
python scripts/check_structure.py
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Full Setup

```bash
python scripts/verify_setup.py
```

---

## Basic Usage

### Train a Probe for a Specific Concept

```bash
# Train a probe for Tense:Past in English
python scripts/train_probes.py \
    --languages English \
    --concepts Tense
```

### Train Probes for All Configured Concepts

```bash
# Processes English, Spanish, Turkish (multi-treebank)
# for Tense, Number, Gender
python scripts/train_probes.py
```

### Find Correlated Top Features

```bash
# Top 100 features for all concepts
python scripts/find_features.py --k 100

# Top 50 features for a specific concept
python scripts/find_features.py \
    --k 50 \
    --concept_key Tense \
    --concept_value Past
```

### Run a Simple Ablation Experiment

```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Tense \
    --concept_value Past \
    --examples examples/tense_past_english.txt \
    --k 10 \
    --mode simple
```

### Run Progressive Ablation

```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Number \
    --concept_value Plur \
    --examples examples/number_plural_english.txt \
    --k 20 \
    --mode progressive
```

### Feature Necessity Test

```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Tense \
    --concept_value Past \
    --examples examples/tense_past_english.txt \
    --k 10 \
    --mode necessity
```

---

## Create Your Own Examples

Create a text file in the following format:

```
positive
Sentence with the concept present.
Another sentence with the concept.

negative
Sentence without the concept.
Another sentence without the concept.
```

Save it in `examples/` and use it with the `--examples` argument.

---

## Add More Languages

1. Edit `src/config.py`:

```python
LANGUAGES = ["English", "Spanish", "Turkish", "French"]
```

2. Make sure the UD treebanks exist in:

```
/projectnb/mcnet/jbrin/.cache/ud/ud-treebanks-v2.16/UD_French-*/
```

The system will automatically find and use all available treebanks.

3. Run the pipeline as usual.

---

## Add More Concepts

Concepts are automatically detected from the UD treebanks. Just add them to the configuration:

```python
CONCEPTS = ["Tense", "Number", "Gender", "Case", "Mood"]
```

---

## View Results

### Trained Probes

```bash
ls /lang-probing/outputs/probes/
# English_Tense_Past.joblib
# Spanish_Tense_Past.joblib
# Turkish_Tense_Past.joblib
# etc.
```

### Identified Features

```bash
ls /lang-probing/outputs/features/
# Tense_Past.json
# Tense_Past_shared.json
# Number_Plur.json
# etc.
```

### Ablation Results

```bash
ls /lang-probing/outputs/ablations/
# English_Tense_Past_simple_k10.json
# Spanish_Number_Plur_progressive_k20.json
# etc.
```

---

## Tips

* **Recommended GPU**: The model requires ~24GB VRAM
* **Batch size**: Reduce `BATCH_SIZE` in `src/config.py` if you encounter memory issues
* **Minimum samples**: Adjust `--min_samples` if a concept has few instances
* **Logs**: Check the `logs/` folder if something fails

---

## Useful Commands

```bash
# List available probes
ls -lh outputs/probes/

# Count how many probes were trained
ls outputs/probes/*.joblib | wc -l

# View shared features for Tense:Past
cat outputs/features/Tense_Past_shared.json | head -20

# Run tests
python -m pytest tests/

# View recent logs
tail -f logs/train_probes.log
```

---

## Quick Troubleshooting

### Error: "No training files found"

UD treebanks do not exist for the specified language. Check:

```bash
ls /project/directory/ud/ud-treebanks-v2.16/UD_English-*/
```

### Error: "Not enough samples"

The concept has too few instances. Try:

* Lowering `--min_samples` (e.g. `--min_samples 64`)
* Using a different concept
* Using another treebank

### Error: "CUDA out of memory"

Reduce the batch size in `src/config.py`:

```python
BATCH_SIZE = 8  # or 4
```

### Error: "Probe already exists"

Use `--overwrite` to retrain:

```bash
python scripts/train_probes.py --overwrite
```

---