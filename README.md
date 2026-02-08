# Language probing -> steering vector activation project

## Description

I'll put more information here later, but so far this has been a project in probing for grammatical concepts in Llama across languages. I'm currently working on learning steering vectors for grammatical concept values across languages and layers. Further work will investigate relations between them, causal verification, and the implications for how language models implement translation.

## Usage

As there are dozens of grammatical concepts and values, we first collect all activations alongside a dictionary of treebank tags for each sentence.

Use `scripts/collect_activations.py` with a list of `--languages` to collect layer outputs (mean-pooled over token positions) for each sentence of that language in Universal Dependencies (`UD_BASE_FOLDER`) and each layer in `COLLECTION_LAYERS`. These outputs will be stored in `ACTIVATIONS_DIR` by `language={Language}/layer={layer_num}/data.parquet`.

The file `scripts/collect_steering_vectors.py` expects activations to be stored in this format.

## Current Project State

### What's Been Implemented

The core infrastructure is in place and most experiments have been run. Here's what's working:

**Activation Collection**: Activations have been collected for 23 languages across 9 layers (0, 4, 8, 12, 16, 20, 24, 28, 31) and stored in Parquet format. The system supports multi-treebank concatenation automatically.

**Probe Training**: Linear probes have been trained for multiple grammatical concepts (Tense, Number, Gender, etc.) across languages. Probes are saved in `outputs/probes/` as `.joblib` files.

**Input Features**: Difference-in-means vectors have been computed for sentence-level input features across multiple languages. These are stored in `outputs/sentence_input_features/{language}/{concept}/{value}/diff_vector.pt`. Both sentence-level and word-level procedures are implemented.

**Output Features**: Attribution patching has been implemented and run extensively to find output features during translation. Results are in `outputs/attributions/` with effects files for various language pairs. The system uses integrated gradients to compute feature attributions from probe logits.

**Ablation Experiments**: The main ablation pipeline is implemented and has been executed. Seven experimental configurations are available:
- Monolingual input/output feature ablation
- Multilingual input/output feature ablation (ablating source/target and measuring target)
- Random baselines for all configurations

Results are saved in `outputs/ablation_results/` as JSONL files. The system measures Δp(reference) - the change in log-probability of the correct reference sequence before and after ablations, normalized as (new - old) / old.

**Feature Analysis**: Code exists for comparing input and output features, finding shared features across languages, and computing Jaccard similarity between feature sets.

### What Still Needs Work

Some visualization and analysis components from the paper outline are still pending:
- Spearman correlation plots for input features (Figure 4 in outline)
- Spearman correlation plots for output features (Figure 5 in outline)  
- Systematic grid of bar charts showing input/output feature overlap by language and k
- Analysis comparing average activations in translation vs monolingual contexts

The BLEU measurement code (`ablate_bleu()`) is partially implemented but needs completion.

### Key Scripts

- `scripts/collect_activations.py` - Collect activations from UD treebanks
- `scripts/train_probes.py` - Train linear probes for concepts
- `scripts/sentence_input_features.py` - Compute input feature vectors (difference-in-means)
- `scripts/attribution_flores.py` - Find output features via attribution patching
- `scripts/ablate.py` - Run ablation experiments (monolingual/multilingual, input/output, with baselines)
- `scripts/input_output_features_visualize.py` - Compare input and output features

### Results Location

- Activations: `outputs/activations/language={Language}/layer={layer}/data.parquet`
- Probes: `outputs/probes/{language}_{concept}_{value}.joblib`
- Input features: `outputs/sentence_input_features/{language}/{concept}/{value}/diff_vector.pt`
- Output features (attributions): `outputs/attributions/{timestamp}/effects_{source}_{target}.pt`
- Ablation results: `outputs/ablation_results/results_{experiment_type}.jsonl`
- Feature analysis: `outputs/features/{concept}_{value}.json` and `{concept}_{value}_shared.json`

## Note

If you download the full UD treebank, keep in mind that the `UD_Arabic-NYUAD` treebank needs to be combined with its data; look at its README for more information.

## Acknowledgements

A significant amount of code for the sentence-level probing experiments is from Jannik Brinkmann's code at https://github.com/jannik-brinkmann/multilingual-features