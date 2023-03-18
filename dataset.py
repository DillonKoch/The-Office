# ==============================================================================
# File: dataset.py
# Project: The-Office
# File Created: Wednesday, 31st December 1969 6:00:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Friday, 17th March 2023 2:14:46 pm
# Modified By: Dillon Koch
# -----
#
# -----
# pytorch dataset class for loading office data
# ==============================================================================

import re
import sys
from collections import Counter
from os.path import abspath, dirname

import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from tqdm import tqdm

from utilities import list_characters

ROOT_PATH = dirname(abspath(__file__))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class OfficeDataset(Dataset):
    def __init__(self, split, sequence_length):
        self.split = split  # * train, val, test
        self.sequence_length = sequence_length
        self.df = pd.read_csv(ROOT_PATH + f"/data/{split}.csv")

        # ! only using some data
        self.df = self.df.iloc[:100, :]

        self.lines, self.characters = self.oversample_lines()
        self.unique_characters = list_characters()
        self.vocab = self.build_vocabulary()
        print("here")

    def oversample_lines(self):  # Top Level __init__
        """
        using the raw lines and speakers from the dataframe, this method
        oversamples lines to make the class distribution even
        """
        lines = list(self.df['line_text'])
        characters = list(self.df['speaker'])

        character_counts = dict(Counter(characters))
        max_count = max(character_counts.values())

        char_line_dict = {outer_character: [line for inner_character, line in zip(characters, lines) if inner_character == outer_character]
                          for outer_character in set(characters)}
        for key in char_line_dict:
            new_list = char_line_dict[key] * 1000
            new_list = new_list[:max_count]
            char_line_dict[key] = new_list

        characters_output = []
        lines_output = []
        for character, lines in char_line_dict.items():
            characters_output += [character] * max_count
            lines_output += lines

        return lines_output, characters_output

    def _clean_text_to_tokens(self, text):
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        tokens = word_tokenize(text)
        # tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [token.lower() for token in tokens]
        # tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def build_vocabulary(self):  # Top Level __init__
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for line in tqdm(list(self.df['line_text'])):
            tokens = self._clean_text_to_tokens(line)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

    def __len__(self):  # Run
        return len(self.lines)

    def __getitem__(self, idx):  # Run
        """
        returns line, character
        - line is a vector of word indices from self.vocab
        - character is the index of the character
        """
        line = self.lines[idx]
        tokens = self._clean_text_to_tokens(line)
        indices = [self.vocab[token] for token in tokens]
        while len(indices) < self.sequence_length:
            indices.append(0)
        indices = indices[:self.sequence_length]

        character = self.characters[idx]
        character_idx = self.unique_characters.index(character)

        return torch.tensor(indices), character_idx


if __name__ == "__main__":
    x = OfficeDataset('train', 100)
    x.__getitem__(0)
