import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def hypothesis1():
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


print('begin')

titanic_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# hypothesis1()

titanic_df = titanic_df.dropna(axis=0)

features = ['Pclass', 'Age', 'Fare']

y = titanic_df.Survived
X = titanic_df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

prediction = model.predict(val_X)
mae = mean_absolute_error(val_y, prediction)
print(mae)

# print(model.predict([[1, 40]]))

print('end')
