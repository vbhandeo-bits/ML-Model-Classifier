# ML Model Classifier - BITS Assignment 2

A Streamlit-based machine learning application for classification tasks. This project implements multiple classification algorithms with evaluation metrics and visualization capabilities.

## What's Included

**Data Management**
- CSV file upload functionality with sample HR Employee Attrition dataset
- Download pre-loaded sample dataset directly from the app for easy testing
- Automatic preprocessing of categorical data using label encoding
- Dataset overview showing record count, features, and missing values
- Data statistics explorer

**Classification Models**
I've implemented 6 different classification algorithms:
1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest
6. XGBoost

You can compare all of them using the dropdown selector.

**Advanced Features**
- Compare all 6 models side-by-side with performance metrics
- Model comparison visualization chart
- Cross-validation analysis (5-fold) with stability metrics
- Feature importance analysis for tree-based models
- Precision-Recall curve for binary classification
- Model reliability assessment with recommendations

**Evaluation Metrics**
The app calculates and displays:
- Accuracy - overall correctness percentage
- Precision - how accurate positive predictions are
- Recall - how many actual positives we catch
- F1 Score - balance between precision and recall
- Matthews Correlation Coefficient (MCC) - good for multi-class problems
- ROC-AUC Score - for binary classification analysis
- Cross-validation scores - ensures model isn't overfitting

**Visualizations**
- Model performance comparison bar chart
- Confusion Matrix as a heatmap to see what the model got right/wrong
- ROC Curve with AUC score to evaluate binary classifier performance
- Precision-Recall curve for threshold analysis
- Feature importance bar chart (for tree models)
- Classification Report with per-class breakdown

## Sample Dataset

The app includes a pre-loaded **HR Employee Attrition** dataset that you can download directly from the application. This dataset contains:
- Employee information and attributes
- Multiple features for classification
- Binary target variable (Attrition: Yes/No)
- Perfect for testing all classification models

Click the "📊 Download Sample Dataset" button in the app to get the CSV file.

## Installation

You need Python 3.8 or higher.

1. Clone or download this project folder

2. Install the required packages:
```bash
pip install -r requirements.txt
```

The main dependencies are:
- streamlit (for the web interface)
- pandas (data handling)
- scikit-learn (ML algorithms and metrics)
- xgboost (gradient boosting)
- matplotlib and seaborn (visualizations)

---

## How to Run

Use this command to start the application:
```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

If you're having issues with streamlit not being found, the command above should work on all systems.

## Using the App

1. Upload a CSV file with your dataset
2. The app shows you basic info about your data (number of rows, columns, missing values)
3. **NEW:** Check "Compare All 6 Models" to see how all algorithms perform side-by-side
4. Select a specific model from the dropdown for detailed analysis
5. Review all metrics, visualizations, and feature importance
6. Check the cross-validation scores to ensure the model isn't overfitting
7. Read the model summary and recommendation at the bottom

## Data Format

Your CSV file should have features in columns and the target variable (what you want to predict) in the **last column**. 

Example:
```
Age,Income,Education,Target
25,35000,HighSchool,Yes
45,75000,Bachelor,Yes
32,50000,Associate,No
```

The app can handle both numeric values and text/categorical data.

## How It Works

When you upload data, the app:
1. Loads your CSV file
2. Converts any text features to numbers (because ML algorithms need numbers)
3. Splits data into 80% training and 20% testing
4. Trains the selected model on the training data
5. Tests it on the test data
6. Calculates all the metrics and creates visualizations

The train-test split uses a fixed random seed (42) so you get the same results every time.

## What Each Metric Means

**Accuracy** - Out of all predictions, how many did the model get right? This is the basic metric but can be misleading with imbalanced data.

**Precision** - When the model predicts positive class, how often is it correct? Important when false positives are bad.

**Recall** - Out of all actual positive cases, how many did the model catch? Important when missing positives is bad.

**F1 Score** - Combines precision and recall. Good when you care about both.

**MCC** - Works well for imbalanced datasets and multi-class problems. Ranges from -1 to 1, where 1 is perfect.

**ROC-AUC** - Shows the model's ability to distinguish between classes (only works for 2-class problems). 0.5 is random, 1.0 is perfect.

## Features I Added

- Handles both binary and multi-class classification automatically
- Supports both numeric and categorical features
- Color-coded heatmaps for confusion matrices
- Professional ROC curves with baseline for comparison
- Data statistics that you can expand to see more details
- All metrics are weighted appropriately for multi-class problems
- **NEW:** Compares all 6 models in a table and chart view
- **NEW:** Cross-validation analysis shows if model is overfitting
- **NEW:** Feature importance for decision trees, random forests, and XGBoost
- **NEW:** Precision-Recall curve for binary classification problems
- **NEW:** Model summary with reliability assessment and recommendations

## Assignment Checklist

Here's what the assignment asked for and what I implemented:

- Data loading from CSV ✓
- Data preprocessing ✓
- Multiple classifier algorithms (6 total) ✓
- Model training ✓
- Evaluation metrics calculation ✓
- Confusion matrix visualization ✓
- Classification metrics display ✓
- ROC curve visualization ✓

All requirements are covered and the code should handle edge cases properly.

## Project Files

```
.
├── app.py              # Main application code
├── requirements.txt    # Package dependencies
├── README.md          # This file
└── ML_Assignment_2.pdf # Assignment instructions
```

## Possible Issues

**Streamlit error / command not found**
- Try: `python -m streamlit run app.py`

**Module not found errors**
- Make sure you ran `pip install -r requirements.txt`

**CSV upload not working**
- Check that your CSV is valid and the target variable is in the last column

**Metrics show errors**
- Make sure your target column has at least 2 different values (for classification)

## What Models Work Best

- **Logistic Regression**: Fast, good for simple problems, interpretable
- **Decision Tree**: Easy to understand what the model is doing
- **KNN**: Simple but can be slow with large datasets
- **Naive Bayes**: Works well when features are independent
- **Random Forest**: Usually gives good results, handles complex relationships
You can try different models on your data to see which works best.

## Notes

This is built with Streamlit which makes it really easy to create interactive ML apps without needing to know web development. The app runs locally so your data never gets sent anywhere.

I used scikit-learn for most algorithms because it's reliable and well-documented. XGBoost is included for the more advanced gradient boosting model.

If you want to test the app, you can use any classification dataset. Good options include the Iris dataset, credit card approval data, or any tabular dataset with a clear target variable.

Hope this helps with the assignment! Let me know if you run into any issues.
