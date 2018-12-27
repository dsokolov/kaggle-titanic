import pandas as pd

import src.decision_tree
import src.io_utils

print('begin')

titanic_train_df = pd.read_csv("../input/train.csv")
titanic_test_df = pd.read_csv("../input/test.csv")

# src.heuristic.run(train_df=titanic_train_df, test_df=titanic_test_df)

src.decision_tree.run(train_df=titanic_train_df, test_df=titanic_test_df)

print('end')
