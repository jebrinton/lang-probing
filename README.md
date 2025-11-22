# Language probing -> steering vector activation project

## Description

I'll put more information here later, but so far this has been a project in probing for grammatical concepts in Llama across languages. I'm currently working on learning steering vectors for grammatical concept values across languages and layers. Further work will investigate relations between them, causal verification, and the implications for how language models implement translation.

## Usage

As there are dozens of grammatical concepts and values, we first collect all activations alongside a dictionary of treebank tags for each sentence.

Use `scripts/collect_activations.py` with a list of `--languages` to collect layer outputs (mean-pooled over token positions) for each sentence of that language in Universal Dependencies (`UD_BASE_FOLDER`) and each layer in `COLLECTION_LAYERS`. These outputs will be stored in `ACTIVATIONS_DIR` by `language={Language}/layer={layer_num}/data.parquet`.

The file `scripts/collect_steering_vectors.py` expects activations to be stored in this format.

## Note

If you download the full UD treebank, keep in mind that the `UD_Arabic-NYUAD` treebank needs to be combined with its data; look at its README for more information.

## Acknowledgements

A significant amount of code for the sentence-level probing experiments is from Jannik Brinkmann's code at https://github.com/jannik-brinkmann/multilingual-features
