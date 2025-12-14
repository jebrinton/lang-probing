# %% Explore the UD output files

import os
import glob
import itertools
import torch
from tqdm import tqdm
import numpy as np

out_dir = "/projectnb/mcnet/jbrin/lang-probing/outputs/attributions"

# Iterate through the subfolders, each corresponding to a language pair
effects_files = {}
for subfolder in tqdm(os.listdir(out_dir)):
    if os.path.isdir(os.path.join(out_dir, subfolder)):

        # Find effects_*.pt file
        effects_file = glob.glob(os.path.join(out_dir, subfolder, "effects_*.pt"))
        if len(effects_file) == 0:
            raise ValueError(f"No effects file found for {subfolder}")
        effects_file = effects_file[0]
    
        # The suffix is a language pair, such as effects_English_French.pt
        source_lang, target_lang = effects_file.split("/")[-1].split("_")[1:]
        target_lang, _ = target_lang.split(".")
        language_pair = (source_lang, target_lang)

        print(f"Loading effects file for {language_pair}")

        # Load the effects file and, if necessary, overwrite with more recent version
        # Make sure they're on CPU
        effects = torch.load(effects_file)
        effects_files[language_pair] = effects

# %% Identify all language pairs and concepts and values

language_pairs = set()
concepts = {}
for language_pair in effects_files:
    language_pairs.add(language_pair)
    for concept_key in effects_files[language_pair]:
        if concept_key not in concepts:
            concepts[concept_key] = set()
        for concept_value in effects_files[language_pair][concept_key]: 
            concepts[concept_key].add(concept_value)

print(f"Language pairs: {language_pairs}")
print(f"Concepts: {concepts}")

# %% Find the top K features for each concept key-value pair
top_features = {}
K = 30
for source_lang, target_lang in language_pairs:

    for concept_key in concepts:
        if concept_key not in effects_files[source_lang, target_lang]:
            continue
        for concept_value in concepts[concept_key]:
            if concept_value not in effects_files[source_lang, target_lang][concept_key]:
                continue

            e = effects_files[source_lang, target_lang][concept_key][concept_value]
            e = e.cpu().numpy()

            avg_effects = np.mean(e, axis=0)

            # Find the top K features that are most positively correlated with the concept and value
            top_features[source_lang, target_lang, concept_key, concept_value] = np.argsort(avg_effects)[-K:]

# %% Barplot where X is a concept key-value pair. Y is the % of features share across at least L languages. Vary L from 2 - max (one barplot for each L).
import matplotlib.pyplot as plt
import numpy as np

# Create a dictionary to count the number of languages across which a feature is present
feature_counts = {}
for concept_key in concepts:
    if concept_key not in feature_counts:
        feature_counts[concept_key] = {}
    for concept_value in concepts[concept_key]:
        if concept_value not in feature_counts[concept_key]:
            feature_counts[concept_key][concept_value] = {}

        for source_lang, target_lang in language_pairs:
            if (source_lang, target_lang, concept_key, concept_value) not in top_features:
                continue
            for feature in top_features[source_lang, target_lang, concept_key, concept_value]:
                if feature not in feature_counts[concept_key][concept_value]:
                    feature_counts[concept_key][concept_value][feature] = 1
                feature_counts[concept_key][concept_value][feature] += 1

print(f"Feature counts: {feature_counts}")

# Create a big figure with subplots for each concept key-value pair. For that, first count the number of concepts and values
subfigures = 0 
for concept_key in concepts:
    for concept_value in concepts[concept_key]:
        subfigures += 1
print(f"Number of subfigures: {subfigures}")

# Create a big figure with subplots for each concept key-value pair
plt.figure(figsize=(20, 20))
i = 1
for concept_key in concepts:
    for concept_value in concepts[concept_key]:

        # x-axis should be the number of languages across which a feature is present
        x_axis = sorted(list(set(feature_counts[concept_key][concept_value].values())))

        # y-axis should be a count of the number of features that are present on that many languages
        y_axis = [len([feature for feature, count in feature_counts[concept_key][concept_value].items() if count == x]) for x in x_axis]

        plt.subplot(subfigures, subfigures, i)
        plt.bar(x_axis, y_axis)
        plt.xlabel("Number of language pairs")
        plt.ylabel("Number of features")
        plt.title(f"{concept_key} {concept_value}")
        i += 1

plt.show()

# %% 
import matplotlib.pyplot as plt
import numpy as np

# Find all unique languages
unique_languages = set()
for source_lang, target_lang in language_pairs:
    unique_languages.add(source_lang)
    unique_languages.add(target_lang)

# Create a dictionary to count the number of languages across which a feature is present
counts = {}
for language in unique_languages:
    counts[language] = {}
    for concept_key in concepts:
        if concept_key not in counts[language]:
            counts[language][concept_key] = {}
        for concept_value in concepts[concept_key]:
            if concept_value not in counts[language][concept_key]:
                counts[language][concept_key][concept_value] = {}

            for source_lang, target_lang in language_pairs:
                if target_lang != language:
                    continue
                if (source_lang, target_lang, concept_key, concept_value) not in top_features:
                    continue
                for feature in top_features[source_lang, target_lang, concept_key, concept_value]:
                    
                    # If the feature is not in the dictionary, add it
                    if feature not in counts[language][concept_key][concept_value]:
                        counts[language][concept_key][concept_value][feature] = 0
                    
                    # Check if this feature is shared with any other pairs where target language is different
                    for other_source_lang, other_target_lang in language_pairs:
                        if other_target_lang == language:
                            continue
                        if (other_source_lang, other_target_lang, concept_key, concept_value) not in top_features:
                            continue
                        for other_feature in top_features[other_source_lang, other_target_lang, concept_key, concept_value]:

                            # For simplicity, we only count the feature if it is shared with exactly one other language
                            if other_feature == feature:
                                counts[language][concept_key][concept_value][feature] = 1

                        # If the feature is shared with exactly one other language, we can stop checking other languages
                        if counts[language][concept_key][concept_value][feature] > 0:
                            break

# counts 
# %%

# For each lanugage, create a bar plot where each concept key-value is a bar that shows the percentage of features shared with other languages

# Create a big figure with subplots for each language. For that, first count the number of languages
subfigures = 0 
for language in unique_languages:
    subfigures += 1
print(f"Number of subfigures: {subfigures}")

# Create a big figure with subplots for each concept key-value pair
plt.figure(figsize=(60, 60))
i = 1
for language in unique_languages:
    data = {}
    # For each concept key-value pair, create a bar plot where the x-axis is the concept key and the y-axis is the percentage of features shared with other languages
    for concept_key in concepts:
        if concept_key not in counts[language]:
            continue
        for concept_value in concepts[concept_key]:
            if concept_value not in counts[language][concept_key]:
                continue
        
            if len(counts[language][concept_key][concept_value]) == 0:
                continue

            # Get the percentage of features shared with other languages
            number_of_features_shared = 0 
            for feature in counts[language][concept_key][concept_value]:
                if counts[language][concept_key][concept_value][feature] > 0:
                    number_of_features_shared += 1
            percentage_shared = number_of_features_shared / len(counts[language][concept_key][concept_value])
            data[f"{concept_key}_{concept_value}"] = percentage_shared

    plt.subplot(subfigures, subfigures, i)
    plt.bar(data.keys(), data.values())
    plt.xlabel("Concept key-value pair")
    plt.ylabel("Percentage of features shared with other languages")
    plt.title(f"{language}")
    i += 1
plt.show()

# %%