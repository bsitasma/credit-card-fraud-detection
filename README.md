# Credit Card Fraud Detection

This project involves building a machine learning model to detect fraudulent transactions in credit card data.

## Dataset

The dataset used for this project is obtained from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset has a total of 284,807 transactions, with 492 fraudulent transactions.

## Files

- `creditcard.csv`: The dataset used for training and testing the model.
- `fraud_detection.ipynb`: The Jupyter notebook containing the data preprocessing, model training, and evaluation code.
- `Credit_Card_Fraud_Detection_Project.docx`: A document explaining the project, data source, implementation, and model evaluation.

## Data Preprocessing

1. Handling Missing Values
2. Feature Imputation
3. Normalization

## Model Training

The RandomForestClassifier from sklearn.ensemble was used for training the model. The dataset was split into training and testing sets with an 80-20 ratio. The Random Forest model was trained on the training set and evaluated on the test set.

## Model Evaluation

The model's performance was evaluated using several metrics: accuracy, precision, recall, and F1 score. Additionally, a confusion matrix was generated to visualize the model's performance.

## Results

- Accuracy: 0.9998091785135006
- Precision: 1.0
- Recall: 0.9166666666666666
- F1 Score: 0.9565217391304348

## Conclusion

This project demonstrates the application of a machine learning model to detect fraudulent transactions in credit card data. By using techniques such as data imputation and normalization, we were able to preprocess the data effectively. The RandomForestClassifier proved to be a robust model for this task, achieving high accuracy, precision, recall, and F1 score. Identifying fraudulent transactions with high precision is crucial for minimizing financial losses and protecting cardholders.
