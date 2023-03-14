# ==============================================================================
# File: split_data.py
# Project: The-Office
# File Created: Tuesday, 14th March 2023 10:12:52 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 14th March 2023 10:12:52 am
# Modified By: Dillon Koch
# -----
#
# -----
# splitting /data/raw.csv into train.csv, val.csv, test.csv
# ==============================================================================


import sys
from os.path import abspath, dirname

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_PATH = dirname(abspath(__file__))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from utilities import list_characters


class Split_Data:
    def __init__(self):
        pass

    def clean_df(self, df):  # Top Level
        # * cleaning names (some characters had multiple names used / typos)
        df['speaker'] = df['speaker'].replace('Micheal', 'Michael')
        df['speaker'] = df['speaker'].replace('Robert California', 'Robert')
        df['speaker'] = df['speaker'].replace('David Wallace', 'David')

        # * keeping only rows with the top 21 characters
        df = df.loc[df['speaker'].isin(list_characters())]
        return df

    def run(self):  # Run
        # * splitting into train/val/test 80-10-10
        df = pd.read_csv(ROOT_PATH + "/data/raw.csv", encoding='unicode_escape')
        df = self.clean_df(df)
        train, holdout = train_test_split(df, test_size=0.2, random_state=18)
        val, test = train_test_split(holdout, test_size=0.5, random_state=18)

        train.to_csv(ROOT_PATH + "/data/train.csv", index=False)
        val.to_csv(ROOT_PATH + "/data/val.csv", index=False)
        test.to_csv(ROOT_PATH + "/data/test.csv", index=False)


if __name__ == '__main__':
    x = Split_Data()
    self = x
    x.run()
