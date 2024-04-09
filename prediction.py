
import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

class Pipeline:
    """
    Pipeline
    =====================
    The Pipeline class provides a way to preprocess and model data for training and testing.

    Methods:
    ---------------------
    run(data: pd.DataFrame, test: bool = False):
        Preprocesses and models the input data.

        Parameters:
        ---------------------
        data (pandas DataFrame): The input data to preprocess and model.
        test (bool): If True, load the model from their saved files and
            use them to predict the output for the input data.
        """

    def __init__(self):
        """
        Initializes the Pipeline class.

        """
        self.model_filename = "model.sav"  # Test mode
        self.target_column = "Dependent-Company Status"
        self.model = DecisionTreeClassifier(class_weight='balanced', random_state=11)

    def run(self, data, test=False):
        """
        Preprocesses and models the input data.

        Parameters:
        ---------------------
        data (pandas DataFrame): The input data to preprocess and model.
        test (bool): If True, load the  model from their saved files and use them
            to predict the output for the input data.

        """
        data.replace({'No Info': ('NaN'), 'unknown amount': ('NaN')}, inplace=True)
        X = data.drop(columns=['Dependent-Company Status'])  # Features
        y = data['Dependent-Company Status']  # Target variable
        X = pd.get_dummies(X, drop_first=True)

        if test:
            # Model and Preprocessor loading process.
            self.model = joblib.load(self.model_filename)

            imputer = SimpleImputer(strategy='most_frequent')
            X = imputer.fit_transform(X)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            y_predict = self.model.predict(X)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            y_predict = self.model.predict(X)
            dt_accuracy = accuracy_score(y, y_predict)
            print('Test Accuracy Score',dt_accuracy )


        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            imputer = SimpleImputer(strategy='most_frequent')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.fit_transform(X_test)

            scaler = StandardScaler()
            X_train_encoded_imputed = scaler.fit_transform(X_train)
            X_test_encoded_imputed = scaler.fit_transform(X_test)

            dt_model = self.model
            dt_model.fit(X_train_encoded_imputed, y_train)
            dt_pred = dt_model.predict(X_test_encoded_imputed)
            dt_accuracy = accuracy_score(y_test, dt_pred)
            print("Test Accuracy after Training is done", dt_accuracy)

def main():
    """
    main()
    --------------------
    The main() function serves as the entry point of the program.

    It uses the argparse module to parse command line arguments and the preprocessor
    module from preprocessor.py to preprocess the data. Additionally,
    it uses the Model module from model.py to train the model.

    If the program is run in test mode, it imports the pre-trained
    model and generates predictions using the imported model.
    """

    # Define argparse.ArgumentParse object for argument manipulating
    parser = argparse.ArgumentParser(
        description="""
        It defines two arguments that can be passed to the program:

        1. "--data_path": a required argument that 
            specifies the path to the data file.

        2. "--inference": an optional argument of type bool 
            that activates test mode if set to "True".
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add --data_path and --inference arguments to parser
    parser.add_argument("--data_path", default ='data.csv', help="Path to data file.", required=False)
    parser.add_argument("--inference", default =False, help="Test mode activation.", required=False)

    # --inference -> default = False. Default activates train mode.

    # Get arguments as dictionary from parser
    args = parser.parse_args()  # returns dictionary-like object

    possible_falses = ["0", "false", "False"]

    path_of_data = args.data_path
    test_mode = args.inference not in possible_falses

    # Reading CSV file
    DataFrame = pd.read_csv(path_of_data, encoding='latin1')

    # Pipeline running
    pipeline = Pipeline()
    pipeline.run(DataFrame, test=test_mode)


if __name__ == "__main__":
    main()
