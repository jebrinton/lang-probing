""""
\item $\Delta p(\text{reference})$: change in (log-)probability of the correct reference sequence before and after ablations. Pre-fill the correct sequence and measure the logits of the correct token at all target language positions
\begin{itemize}
    \item Translation: give a 2-shot example
    \item Non-translation: no examples, single sentence
    \item Normalize by the original probability (new - old) / old
    \item Try this with input and output features
    \item Also do a random baseline
    \item [2nd] Input features translation: ablate features at source sentence token indices, measure prob of target sequence before and after ablation
    \item Input features monolingual: ablate features for sentence, measure prob of sentence before and after ablation
    \item Output features translation: ablate features at target sequence token indices, measure prob of target sequence before and after ablation
    % possible for a model to perform grammatical computations before we start the ablation (during the source sequence), then copy it over in layers after the 16th layer and it won't get ablated
    \item Output features monolingual: ablate features for sentence, measure prob of sequence before and after ablation
"""

from pathlib import Path
import itertools
import logging

from nnsight import LanguageModel
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np

from lang_probing_src.utils import setup_model
from lang_probing_src.config import MODEL_ID, SAE_ID, NAME_TO_LANG_CODE, LANGUAGES_DEC
from lang_probing_src.ablate import ablate
from lang_probing_src.utils_input_output import get_output_features_vector, get_input_features_vector, load_effects_files, get_language_pairs_and_concepts


def get_feature_indices(K, feature_vector):
    """Return the top K feature indices from a feature vector"""
    return np.argsort(feature_vector)[-K:] # get the largest K features


def get_positions(tokenizer, context, source_sentence, target_sentence):
    # 1. Create the full prompt
    full_text = context + source_sentence + target_sentence

    # 2. Find character start/end of the Spanish substring
    # Note: You must ensure 'source_sentence' is exactly as it appears in full_text
    source_sentence_start = len(context)
    target_sentence_start = source_sentence_start + len(source_sentence)
    full_text_end = target_sentence_start + len(target_sentence)

    # 3. Tokenize with offsets
    encoding = tokenizer(full_text, return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"] # List of (char_start, char_end) tuples
    input_ids = encoding["input_ids"]

    # 4. Find which tokens fall within the character range
    target_token_indices = []
    source_token_indices = []

    for idx, (start, end) in enumerate(offset_mapping):
        # We check if the token has significant overlap with the spanish substring
        # Using max(start, char_start) < min(end, char_end) checks for ANY overlap
        if start >= source_sentence_start and end <= target_sentence_start:
            source_token_indices.append(idx)
        elif start >= target_sentence_start and end <= full_text_end:
            target_token_indices.append(idx)

    # print the tokens at ablate_positions and prob_positions

    ablate_positions = slice(source_token_indices[0], source_token_indices[-1])
    prob_positions = slice(target_token_indices[0], target_token_indices[-1])

    # print the whole prompt
    
    # Print the tokens at ablate_positions and prob_positions
    ablate_tokens = tokenizer.decode(input_ids[ablate_positions], skip_special_tokens=True)
    prob_tokens = tokenizer.decode(input_ids[prob_positions], skip_special_tokens=True)

    logging.info(f"Context: {context}")

    # logging.debug("Ablate positions indices:", list(range(ablate_positions.start, ablate_positions.stop)))
    logging.info("Ablate position tokens:", ablate_tokens)
    # logging.debug("Prob positions indices:", list(range(prob_positions.start, prob_positions.stop)))
    logging.info("Prob position tokens:", prob_tokens)

    logging.info(f"Source sentence: {source_sentence}")
    logging.info(f"Target sentence: {target_sentence}")

    exit()

    return input_ids, ablate_positions, prob_positions

def main():
    logging.basicConfig(level=logging.INFO)
    max_num_samples = 100
    model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
    input_features_dir = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features")
    effects_files = load_effects_files()

    concept = "Tense"
    value = "Past"
    for source_lang, target_lang in itertools.permutations(LANGUAGES_DEC, 2):
        if source_lang == "Chinese":
            source_lang = "Chinese (Simplified)"
        if target_lang == "Chinese":
            target_lang = "Chinese (Simplified)"
        source_dataset = load_dataset("gsarti/flores_101", NAME_TO_LANG_CODE[source_lang], split="devtest")
        target_dataset = load_dataset("gsarti/flores_101", NAME_TO_LANG_CODE[target_lang], split="devtest")

        input_features = get_input_features_vector(input_features_dir, source_lang, concept, value)
        output_features = get_output_features_vector(effects_files, (source_lang, target_lang), concept, value)

        K = 30
        feature_indices = np.argsort(output_features)[-K:]
        random_feature_indices = np.random.choice(len(output_features), K, replace=False)

        num_samples = min(len(source_dataset), max_num_samples)
        for i in range(num_samples):
            context = f"""
            {source_dataset[(i+2)%num_samples]['sentence']} >> {target_dataset[(i+2)%num_samples]['sentence']}
            {source_dataset[(i+1)%num_samples]['sentence']} >> {target_dataset[(i+2)%num_samples]['sentence']}            
            """
            source_sentence = source_dataset[i]['sentence']
            target_sentence = target_dataset[i]['sentence']
            input_ids, ablate_positions, prob_positions = get_positions(tokenizer, context, source_sentence, target_sentence)

            delta_p = ablate(
                model,
                submodule,
                autoencoder,
                tokenizer,
                input_ids=input_ids,
                feature_indices=feature_indices,
                ablate_positions=ablate_positions,
                prob_positions=prob_positions
            )

            mean_delta_p = delta_p.mean()
            min_delta_p = delta_p.min()

            print(f"Mean delta p: {mean_delta_p}")
            print(f"Min delta p: {min_delta_p}")

if __name__ == "__main__":
    main()
