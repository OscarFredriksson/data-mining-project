import pandas as pd

import dataset_utility as du
import decision_tree as dt
import naive_bayes as nb

df = pd.read_csv("mushrooms-filled-missing.csv", names=du.get_column_names())

print("---Decision Tree--")
dt.do_decision_tree(df)

print("\n---Naive Bayes---")
nb.do_naive_bayes(df)
