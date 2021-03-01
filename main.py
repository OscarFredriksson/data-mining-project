import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time

from preprocess import *

column_names = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

df = pd.read_csv("mushrooms.csv", names=column_names)

train, missing = extract_missing(df)

train = pd.DataFrame(train, columns=column_names)
missing = pd.DataFrame(missing, columns=column_names)

print("TRAIN:")
print(train["stalk-root"])

print("\nMiSSING:")
print(missing["stalk-root"])