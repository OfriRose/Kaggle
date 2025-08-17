import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def display_data(train_df, test_df):
    # Display the first few rows of the training data
    print("Training Data Head:")
    print(train_df.head())

    # Get information about the columns (data types and non-null counts)
    print("\nTraining Data Info:")
    train_df.info()

    # Display the first few rows of the test data
    print("\nTest Data Head:")
    print(test_df.head())

    # Get information about the columns
    print("\nTest Data Info:")
    test_df.info()


def main():
    train_df = pd.read_csv('./data/train.csv')
    test_df  = pd.read_csv('./data/test.csv')    
#    display_data(train_df= train_df, test_df=test_df)

    #########data preprocessing#########

    #replacing missing age values with the median
    train_df['Age'].fillna(train_df['Age'].median())
    test_df['Age'].fillna(test_df['Age'].median())

    #replacing missing embarked values with the mode
    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
    test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

    #replacing missing fare values with the median
    train_df['Fare'].fillna(train_df['Fare'].median())
    test_df['Fare'].fillna(test_df['Fare'].median())


    #converting categorical variables to numerical
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

    train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Embarked', drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Embarked'], prefix='Embarked', drop_first=True)


    #display_data(train_df=train_df, test_df=test_df)

    #########model training#########

    #define features and target variable
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_S', 'Embarked_Q']
    target = 'Survived'
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]

    model = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=42)
    model.fit(X_train, y_train)

    ###########model prediction#########

    predictions = model.predict(X_test)


    ###########output results#########
    submission_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
    })
    
    submission_df.to_csv('titanic_submission.csv', index=False)

if __name__ == "__main__":
    main()

