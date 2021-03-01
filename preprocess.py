def extract_missing(dataset):

    train = []
    missing = []

    for index, row in dataset.iterrows():
        if row["stalk-root"] == "?":
            missing.append(row)
        else:
            train.append(row)

    return train, missing