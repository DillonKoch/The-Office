# ==============================================================================
# File: utilities.py
# Project: The-Office
# File Created: Tuesday, 14th March 2023 10:34:03 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 14th March 2023 10:34:05 am
# Modified By: Dillon Koch
# -----
#
# -----
#
# ==============================================================================


def list_characters():
    """
    top 21 characters used for classification
    """
    return ["Michael",
            "Dwight",
            "Jim",
            "Pam",
            "Andy",
            "Kevin",
            "Angela",
            "Oscar",
            "Erin",
            "Ryan",
            "Darryl",
            "Phyllis",
            "Kelly",
            "Jan",
            "Toby",
            "Stanley",
            "Meredith",
            "Robert",
            "David",
            "Karen",
            "Creed"]


def char_to_idx_dict():
    """
    mapping character names to their corresponding idx for predictions
    """
    return {character: i for i, character in enumerate(list_characters())}


def idx_to_char_dict():
    """
    mapping indices to character names
    """
    return {i: character for character, i in char_to_idx_dict.items()}
