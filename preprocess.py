def extract_missing(dataset):

    train = []
    missing = []

    for index, row in dataset.iterrows():
        if row["stalk-root"] == "?":
            missing.append(row)
        else:
            train.append(row)

    return train, missing


def split_features_labels(dataset):
    features = dataset.copy()

    features.pop("class")

    labels = dataset[["class"]].copy()

    return features, labels