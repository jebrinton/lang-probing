# get number of tokens in all conllu files in the config.LANGUAGES

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyconll
from src.config import LANGUAGES, MODEL_ID
from src.data import get_all_treebank_files
from src.utils import setup_model
from tqdm import tqdm

# model, _, _, tokenizer = setup_model(MODEL_ID)

# total_tokens = {language: 0 for language in LANGUAGES}
# for language in LANGUAGES:
#     files = get_all_treebank_files(language, 'train')
#     for file in files:
#         data = pyconll.load.load_from_file(file)
#         with tqdm(total=len(data), desc=f"Processing {language}") as pbar:
#             for sentence in data:
#                 input_ids = tokenizer(sentence.text, return_tensors="pt")["input_ids"]
#                 total_tokens[language] += len(input_ids[0])
#                 pbar.update(1)

# print(total_tokens)

total_sentences = {language: 0 for language in LANGUAGES}
for language in LANGUAGES:
    files = get_all_treebank_files(language, 'train')
    for file in files:
        data = pyconll.load.load_from_file(file)
        total_sentences[language] += len(data)

print(total_sentences)