# ==============================================================================
# File: baseline_random.py
# Project: The-Office
# File Created: Tuesday, 14th March 2023 11:17:32 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 14th March 2023 11:17:35 am
# Modified By: Dillon Koch
# -----
#
# -----
# predicting randomly as a baseline
# ==============================================================================


import random
import sys
from os.path import abspath, dirname

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Baseline_Random:
    def __init__(self):
        pass

    def predict(self, text: str):  # Run
        """
        returns an int 0-20 that represents a random character
        """
        return random.choice(list(range(21)))


if __name__ == '__main__':
    x = Baseline_Random()
    self = x
    x.run()
