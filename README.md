# Text Classification Using Support Vector Machines (SVM)

This Python script builds a machine learning pipeline to classify text data using a Support Vector Machines (SVM) classifier. It processes text data, trains a model, and evaluates its performance on test data, providing accuracy scores and a confusion matrix for visualization.

## Features
- **Text Preprocessing**:
  - Converts raw text data into a numerical format using `CountVectorizer`.
  - Removes stopwords and ignores tokens with low frequency (`min_df=10`).
- **Model Training**:
  - Trains a Support Vector Machines (SVM) classifier on vectorized text data.
- **Model Evaluation**:
  - Predicts labels for test data using the trained model.
  - Calculates accuracy and displays it as both a numeric value and percentage.
  - Generates a confusion matrix for classification results visualization.

## Requirements
- Python 3.6 or later
- Required Python libraries:
  - `pandas` (install via `pip install pandas`)
  - `scikit-learn` (install via `pip install scikit-learn`)
  - `scikit-plot` (install via `pip install scikit-plot`)
  - `matplotlib` (install via `pip install matplotlib`)

## Setup and Usage
1. **Prepare the Dataset**:
   - Create two CSV files:
     - `train_file.csv`: Contains the training data with two columns:
       - `text`: The text data for training.
       - `label`: The corresponding labels.
     - `test_file.csv`: Contains the test data with the same format.
   - Ensure the CSV files are in the same directory as the script.
2. **Install Required Libraries**:
   ```bash
   pip install pandas scikit-learn scikit-plot matplotlib
3. **Run the Script**:
    - Save the script as a Python file (e.g., `text_classification.py`).
    - Execute the script:
    ```bash
    python text_classification.py

4. **Output**:
    - The script prints the model's accuracy on the test dataset.
    - Displays a confusion matrix visualization.

 ## Example Output
```text
The accuracy of this model is: 0.92
The accuracy percentage is: 92.00%
```
  A confusion matrix will also be displayed as a plot.
  
## Notes
**CountVectorizer**:
  - Transforms text data into a sparse matrix of token counts. The parameter min_df=10 filters out words that appear in fewer than 10 documents, reducing noise in the dataset.
    
**Support Vector Machines (SVM)**:
  - The SVM classifier is used for its effectiveness in high-dimensional spaces, making it suitable for text classification tasks.
    
**Confusion Matrix**:
  - The confusion matrix is plotted using `scikit-plot` to visualize the classification performance.

## Customization
**To adjust the preprocessing or model**:
  - Modify the `CountVectorizer` parameters (e.g., `min_df`, `stop_words`).
  - Replace `SVC()` with another classifier from `sklearn` if needed.
    
**To use more classes, update the `class_names` list in the confusion matrix section to include all label names.**
