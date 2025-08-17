Kaggle Titanic Competition: Titanic Data Set
public score: 0.784

My submission implements a solution the scikit-learn's RandomForestClassifier. The goal is to predict which passengers survived the shipwreck based on a given set of features.

Files

    train.csv: The training dataset with survival labels.

    test.csv: The test dataset for which survival predictions must be made.

    titanic_solution.py: The Python script containing the data preprocessing, model training, and prediction logic.

    titanic_submission.csv: The final submission file in the format required by Kaggle.

Method

    Data Loading & Cleaning: Loads the train.csv and test.csv files into pandas DataFrames. Missing values in Age, Fare, and Embarked are handled using imputation (median for numerical, mode for categorical).

    Feature Engineering: Categorical features like Sex and Embarked are converted into numerical format. Sex is mapped to a binary representation (0 or 1), while Embarked is one-hot encoded to create binary columns (Embarked_Q, Embarked_S). Unnecessary columns like Name, Ticket, and Cabin are dropped.

    Model Training: A RandomForestClassifier is initialized and trained on the preprocessed training data. The model learns the relationships between the passenger features and their survival status.

    Prediction: The trained model is used to predict survival outcomes for the passengers in the test dataset.

    Submission: The predictions are formatted into a titanic_submission.csv file, ready to be uploaded to Kaggle.

How to Run

    Place the train.csv and test.csv files in a ./data directory.

    Run the titanic_solution.py script.

    The titanic_submission.csv file will be generated in the same directory.