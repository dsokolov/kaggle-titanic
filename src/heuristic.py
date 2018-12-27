import pandas as pd


def run(train_df, test_df):
    if isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame):
        result_df = hypothesis1(train_df, train_df)
        result_df.to_csv("../output/submission_heuristic.csv", sep=",", encoding="utf-8", index=False)
    else:
        print("Both train_df and test_df should by DataFrame")


def hypothesis1(train_df, test_df):
    df = pd.DataFrame(
        columns=['PassengerId', 'Survived']
    )
    for index, row in test_df.iterrows():
        passenger_id = row['PassengerId']
        gender = row['Sex']
        age = row['Age']
        is_female = gender == 'female'
        is_child = age <= 12
        is_survived = is_female or is_child
        df = df.append(
            pd.DataFrame(
                columns=['PassengerId', 'Survived'],
                data={
                    'PassengerId': [passenger_id],
                    'Survived': [int(is_survived)]
                }
            ), ignore_index=True
        )
    return df
