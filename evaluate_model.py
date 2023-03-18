# ==============================================================================
# File: evaluate_model.py
# Project: The-Office
# File Created: Tuesday, 14th March 2023 10:30:36 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 14th March 2023 10:30:36 am
# Modified By: Dillon Koch
# -----
#
# -----
# evaluating a model's predictions based on F1 score
# ==============================================================================


import sys
from os.path import abspath, dirname

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT_PATH = dirname(abspath(__file__))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from utilities import char_to_idx_dict
from baseline_michael import Baseline_Michael
from baseline_random import Baseline_Random


class Evaluate_Model:
    def __init__(self):
        self.char_to_idx_dict = char_to_idx_dict()

    def run(self, dataset: str, model):  # Run
        """
        - dataset specifies which set to evaluate the model against (train, val, test)
        - model is a model class with a "predict" method
        """
        path = ROOT_PATH + f"/data/{dataset}.csv"
        df = pd.read_csv(path)

        X = list(df['line_text'])
        y = [self.char_to_idx_dict[character] for character in list(df['speaker'])]
        preds = [model.predict(text) for text in X]

        f1 = f1_score(y, preds, average='micro')
        accuracy = accuracy_score(y, preds)
        print(f"F1 Score: {f1}")
        print(f"Accuracy: {accuracy}")


if __name__ == '__main__':
    x = Evaluate_Model()
    self = x

    # model = Baseline_Michael()
    model = Baseline_Random()
    dataset = 'test'
    x.run(dataset, model)
