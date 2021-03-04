def get_column_names():
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

    return column_names


def get_split_column_names(dataset, class_columns):

    feature_columns = dataset.columns.tolist()

    for class_column in class_columns:
        if class_column in feature_columns:
            feature_columns.remove(class_column)

    return class_columns, feature_columns