import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time


# def one_hot_encode(dataset):
#     return pd.get_dummies(dataset, columns=dataset.columns)


def extract_missing(dataset):

    train = []
    missing = []

    for index, row in dataset.iterrows():
        if row["stalk-root"] == "?":
            # print("missing")
            missing.append(row)
        else:
            train.append(row)

    return train, missing


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

print(train)

print(missing)