# NovelMate — Project Report

## Objective
Build a full-stack machine-learning application that predicts a novel's average reader rating and rating category. Users can select one algorithm or compare the predictions of all available models through a web interface.

## Dataset and features
The project uses the Goodbooks-10k `books.csv` dataset. All 10,000 rows passed the target-rating cleaning rules. The six input features are publication year, edition count, ratings count, work ratings count, work text-review count, and language. The regression target is `average_rating`; the classification target is derived as Low, Medium, and High rating groups.

Rating distribution columns (`ratings_1` through `ratings_5`) are excluded to prevent target leakage.

## Preprocessing
The pipeline handles missing numeric values with median imputation, applies a log transformation and standardization to skewed count fields, standardizes publication year, and one-hot encodes language. The preprocessing steps are fitted on training data as part of each scikit-learn Pipeline.

## Models
Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Support Vector Regression.

Classification: Logistic Regression, Decision Tree with Gini, Decision Tree with Entropy, Random Forest, Support Vector Machine, and Gaussian Naive Bayes.

## Evaluation
A held-out test set is used. Regression is evaluated with R², MAE, and RMSE. Classification is evaluated with accuracy and weighted F1-score. Exact scores are saved in `models/training_summary.json` and displayed in the web app.

## Application architecture
- `train_models.py`: cleans the dataset, builds preprocessing pipelines, trains and evaluates the models, and saves the artifacts.
- `app.py`: FastAPI backend that validates input, loads the saved models, and returns predictions.
- `static/index.html`: user interface structure.
- `static/css/style.css`: responsive styling.
- `static/js/app.js`: sends user input to the backend and displays predictions and model comparisons.
- `NovelMate_Colab.ipynb`: runs training and launches the application in Google Colab.

## Limitations
Only book metadata is used. The models cannot interpret a novel's story, genre, writing style, or sentiment. Model scores are therefore modest and predictions should be treated as estimates.
