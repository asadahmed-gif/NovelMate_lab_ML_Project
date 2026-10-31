# NovelMate — Full-Stack Machine Learning

NovelMate is an academic full-stack machine-learning application that predicts a book's average Goodreads rating and classifies it into a **Low**, **Medium**, or **High** rating group. It includes data preprocessing, 10 trained ML models, a FastAPI backend, and a responsive HTML/CSS/JavaScript frontend.

## Project structure

```text
NovelMate_FullStack_ML/
├── data/
│   ├── books.csv                 # Goodbooks-10k dataset
│   └── books_clean.csv           # Generated during training
├── models/
│   ├── regression_*.joblib       # 4 trained regression pipelines
│   ├── classification_*.joblib  # 6 trained classification pipelines
│   ├── model_info.joblib         # Model names, features and test metrics
│   └── training_summary.json     # Readable copy of model information
├── static/
│   ├── index.html                # Frontend page
│   ├── css/style.css             # Website styling
│   └── js/app.js                 # Frontend behaviour and API calls
├── NovelMate_Colab.ipynb         # Main Google Colab notebook
├── train_models.py               # Data preparation and model training
├── app.py                        # FastAPI backend
├── requirements.txt              # Python dependencies
└── README.md
```

## 1. Run in Google Colab

1. Upload this complete project to your public GitHub repository:
   `https://github.com/asadahmed-gif/NovelMate_FullStack_ML`
2. Open `NovelMate_Colab.ipynb` on GitHub.
3. Select **Open in Colab** (or open Colab and use File → Open notebook → GitHub).
4. Run the notebook cells from top to bottom.
5. The notebook installs dependencies, trains all 10 models, starts the backend, and displays the website inside Colab.
6. Use the form to try one model or compare every model.

The notebook clones the GitHub repository, so push the latest project files to GitHub before running it.

## 2. Run on your computer

Use Python 3.10 or newer. Download/clone the repository, open a terminal in the project folder, and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install libraries and train the models:

```bash
pip install -r requirements.txt
python train_models.py
```

Start the website:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser. API documentation is available at `http://127.0.0.1:8000/docs`.

## 3. Machine-learning tasks and models

### Regression — predict average rating (0 to 5)

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR)

Evaluation: **R²**, **MAE**, and **RMSE**.

### Classification — predict rating group

- Logistic Regression
- Decision Tree (Gini)
- Decision Tree (Entropy)
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

Evaluation: **Accuracy** and **weighted F1-score**.

## 4. Data preparation

The project uses the Goodbooks-10k `books.csv` dataset. The target for regression is `average_rating`. The classification target is created from three rating ranges using the one-third and two-third quantiles.

Features used:

- `original_publication_year`
- `books_count`
- `ratings_count`
- `work_ratings_count`
- `work_text_reviews_count`
- `language_code`

Preprocessing is included inside each scikit-learn Pipeline:

- Missing numeric values are filled with the median.
- Negative count values are treated as missing.
- Skewed count values are log-transformed and standardized.
- Publication year is imputed and standardized.
- Language is imputed and one-hot encoded.
- A fixed random seed is used for reproducibility.

The columns `ratings_1` to `ratings_5` are intentionally excluded because they directly reveal rating information and would cause target leakage.

## 5. Backend API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend status |
| GET | `/api/info` | Get model names and evaluation scores |
| POST | `/api/predict/regression` | Predict rating with one regression model |
| POST | `/api/predict/classification` | Predict category with one classifier |
| POST | `/api/compare/regression` | Compare all regression models |
| POST | `/api/compare/classification` | Compare all classification models |

The frontend sends requests to these endpoints using JavaScript `fetch()`.

## 6. Important notes

- GitHub stores and versions the project files; it does not automatically run the Python backend. Use Google Colab or a Python server to run the application.
- If you change the dataset or training code, run the training notebook/script again so the model files and metrics are updated.
- The models use book metadata only. They do not understand plot, genre, writing quality, or reader sentiment, so predictions are estimates rather than guaranteed ratings.
- This is an educational project, not a production book-rating service.

## Dataset

Goodbooks-10k dataset: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k
