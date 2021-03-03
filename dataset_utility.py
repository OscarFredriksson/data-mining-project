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


def get_class_column_names():
    class_columns = [
        "stalk-root_b",
        "stalk-root_c",
        "stalk-root_e",
        "stalk-root_r",
        # "stalk-root_?",
    ]

    return class_columns


def get_split_column_names(dataset):

    feature_columns = dataset.columns.tolist()

    class_columns = get_class_column_names()

    for class_column in class_columns:
        if class_column in feature_columns:
            feature_columns.remove(class_column)

    return class_columns, feature_columns
