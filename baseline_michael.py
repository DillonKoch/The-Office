# ==============================================================================
# File: baseline_michael.py
# Project: The-Office
# File Created: Tuesday, 14th March 2023 10:58:36 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 14th March 2023 10:58:36 am
# Modified By: Dillon Koch
# -----
#
# -----
# predicting "Michael" every time as a baseline
# ==============================================================================

from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(abspath(__file__))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from utilities import char_to_idx_dict


class Baseline_Michael:
    def __init__(self):
        self.char_to_idx_dict = char_to_idx_dict()
        self.michael_idx = self.char_to_idx_dict["Michael"]

    def predict(self, text: str):  # Run
        return self.michael_idx


if __name__ == '__main__':
    x = Baseline_Michael()
    self = x
    # x.run()
