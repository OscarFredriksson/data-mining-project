import pandas as pd

import dataset_utility as du
import decision_tree as dt


df = pd.read_csv("mushrooms-filled-missing.csv", names=du.get_column_names())

dt.do_decision_tree(df)
