
# Startup Company Success and Fail Prediction 



## Files

- `data.csv`: Dataset containing the data for analysis.
- `data_analysis.ipynb`: Jupyter Notebook file for data analysis.
- `model.sav`: Pre-trained machine learning model file.
- `prediction.py`: Python script for making predictions using the pre-trained model.
## data Analysis 

  After conducting data analysis, it was evident that the dataset exhibited class imbalance, with successful outcomes occurring twice as frequently as failures in the target variable (y). Certain features were deemed insignificant for predictive modeling, and filling missing values with "NaN" outperformed other strategies. Furthermore, some features with extensive descriptions posed challenges for effective utilization. Even after attempting to separate investor names, no significant improvement in prediction accuracy was observed. Exploring alternatives, such as breaking down lengthy descriptions into more manageable components, can show promise for enhancing model performance which is a more advanced method. Notably, filling missing values with the 'most frequent' occurrences proved superior to using the 'mean'. These findings underscore the importance of meticulous data preprocessing and feature engineering to optimize predictive modeling outcomes, especially in the context of imbalanced datasets and intricate feature structures.


## Usage Instructions

1. **Clone the Repository**: Clone this Git repository to your local machine using Git commands or a Git client.

2. **Navigate to the Repository Directory**: Open your command-line interface (CLI) and navigate to the directory where you cloned the repository.

3. **Run the Script**: Execute the `prediction.py` script with the following command:
   ```
   python prediction.py --data_path data.csv --inference True
   ```
   - Replace `data.csv` with the actual filename of your dataset if it's named differently.
   - The `--inference True` flag indicates that you want to run the script in test mode, using the pre-trained model to make predictions on your dataset.

4. **View Output**: The script will process your data, load the pre-trained model, make predictions, and print the test accuracy score.

Ensure that your dataset has the required column names and format expected by the script for proper execution. Additionally, update any paths or filenames in the script if they differ from what's specified.
