import pandas as pd

print('begin')

titanic_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

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

df.to_csv("../output/submission.csv", sep=",", encoding="utf-8", index=False)

print('end')
