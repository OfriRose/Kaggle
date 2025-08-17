import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def prepare_data(df):
    """
    Prepares and cleans the DataFrame for model training.
    Handles missing values, feature engineering, and one-hot encoding.
    """
    # Create lists of features to process
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

    # Impute missing numerical values using the median and reassign
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].median())

    # Impute missing categorical values using the mode and reassign
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Deconstruct the 'Cabin' feature and add new columns to categorical features
    df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].str.split('/', expand=True)
    categorical_features.extend(['CabinDeck', 'CabinSide'])
    
    # Drop unnecessary columns
    df.drop(['Cabin', 'Name', 'CabinNum'], axis=1, inplace=True)
    
    return df, categorical_features

def main():
    # Load the training and test datasets
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv') 

    # Save PassengerId for the submission file
    test_ids = test_df['PassengerId']

    # Apply data preparation to both datasets
    train_df_processed, categorical_features = prepare_data(train_df)
    test_df_processed, _ = prepare_data(test_df)

    # Separate features and target from the training set
    y_train = train_df_processed['Transported']
    X_train = train_df_processed.drop('Transported', axis=1)

    # Convert categorical features to numerical using one-hot encoding
    X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
    test_df_processed = pd.get_dummies(test_df_processed, columns=categorical_features, drop_first=True)
    
    # Align the columns of the training and test sets
    X_train, X_test = X_train.align(test_df_processed, join='outer', axis=1, fill_value=0)

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Create and save the submission file
    submission_df = pd.DataFrame({'PassengerId': test_ids, 'Transported': predictions.astype(bool)})
    submission_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()