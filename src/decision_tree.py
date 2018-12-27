import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def run(train_df, test_df):
    print("start decision tree")
    if isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame):
        clear_train_df = clear(train_df)
        X = extract_features(clear_train_df)
        y = extract_y(clear_train_df)
        train_X, val_X, train_y, val_y = split_x_y(X, y)
        best_tree_size = get_best_tree_size(train_X, val_X, train_y, val_y)

        clear_test_df = test_df.fillna(0)
        final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
        final_model.fit(X, y)
        print(final_model)
        X1 = extract_features(clear_test_df)
        y1 = final_model.predict(X1)
        y2 = [int(round(f)) for f in y1]
        result_df = pd.DataFrame(
            columns=['PassengerId', 'Survived'],
            data={
                'PassengerId': test_df.PassengerId,
                'Survived': y2
            }
        )
        result_df.to_csv("../output/submission_decision_tree.csv", sep=",", encoding="utf-8", index=False)
    else:
        print("Both train_df and test_df should by DataFrame")
    print("finish decision tree")


def clear(train_df):
    return train_df.dropna(axis=0)


def extract_features(df):
    X = pd.DataFrame(
        columns=['Age', 'is_male', 'is_female'],
        data={
            'Age': df['Age'],
            'is_male': df['Sex'].map(lambda s: 1 if s == "male" else 0),
            'is_female': df['Sex'].map(lambda s: 1 if s == "female" else 0)
        }
    )
    return X


def extract_y(train_df):
    return train_df.Survived


def split_x_y(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, train_size=0.25)
    return train_X, val_X, train_y, val_y


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


def get_best_tree_size(train_X, val_X, train_y, val_y):
    candidate_max_leaf_nodes = [2, 5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 50, 100]
    scores = {
        leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y)
        for leaf_size in candidate_max_leaf_nodes
    }
    print("scores =", scores)
    best_tree_size = min(scores, key=scores.get)
    print("best tree size", best_tree_size)
    return best_tree_size
