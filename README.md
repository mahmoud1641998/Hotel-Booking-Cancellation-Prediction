# Hotel Booking Cancellation Prediction System

An end-to-end Machine Learning project that predicts hotel booking cancellations using a tuned Random Forest classification pipeline, with an interactive Streamlit web application for predictions and data insights.

## Project Overview

The **Hotel Booking Cancellation Prediction System** is a supervised Machine Learning project designed to predict whether a hotel booking will be canceled or not.

The project uses historical hotel booking records to identify patterns associated with booking cancellations and develop a binary classification model.

Multiple classification algorithms were implemented and evaluated. After model comparison, cross-validation, and hyperparameter tuning using `RandomizedSearchCV`, a **Tuned Random Forest Pipeline** was selected as the final model.

The final model is integrated into an interactive Streamlit application that provides:

- Booking cancellation predictions.
- Cancellation and non-cancellation probabilities.
- Interactive insights and visualizations from the booking dataset.

The project demonstrates a complete Machine Learning workflow, from data exploration and preprocessing to model development, evaluation, automated testing, and web application development.

## Project Objectives

- Explore and understand hotel booking data.
- Identify patterns and relationships associated with booking cancellations.
- Prepare numerical and categorical features for Machine Learning.
- Build and compare multiple classification models.
- Evaluate models using appropriate classification metrics and cross-validation.
- Optimize the selected model using hyperparameter tuning.
- Develop reusable preprocessing and prediction modules.
- Build an interactive web application using Streamlit.
- Implement automated tests to verify core project functionality.

## Dataset Description

The dataset contains **36,285 booking records** and **17 columns**.

| Property | Description |
|---|---|
| Total Records | 36,285 |
| Total Columns | 17 |
| Numerical Features Used | 11 |
| Categorical Features Used | 3 |
| Target Variable | `booking status` |
| Problem Type | Binary Classification |

### Target Variable

The target variable is `booking status`, which contains two classes:

| Class | Description | Records | Percentage |
|---|---|---:|---:|
| Not_Canceled | Booking was not canceled | 24,396 | 67.23% |
| Canceled | Booking was canceled | 11,889 | 32.77% |

### Features Used

**Numerical Features**

- `number of adults`
- `number of children`
- `number of weekend nights`
- `number of week nights`
- `car parking space`
- `lead time`
- `repeated`
- `P-C`
- `P-not-C`
- `average price`
- `special requests`

**Categorical Features**

- `type of meal`
- `room type`
- `market segment type`

### Excluded Columns

- `Booking_ID`: Excluded because it is an identifier, not a meaningful predictive feature.
- `date of reservation`: Excluded from model training because its meaning and availability at prediction time were not sufficiently established.
- `booking status`: Used as the target variable, not as an input feature.

The final model uses 14 input features.

## Project Workflow

The project follows these main stages:

1. **Data Loading and Understanding**
   - Load the dataset.
   - Inspect its structure, data types, and distributions.
   - Examine missing values and duplicate records.

2. **Exploratory Data Analysis (EDA)**
   - Analyze numerical and categorical features.
   - Investigate feature relationships with the target variable.
   - Perform correlation analysis.
   - Review unusual values and feature distributions.

3. **Data Preprocessing**
   - Split the data into training and testing sets using stratified sampling.
   - Apply One-Hot Encoding to categorical features.
   - Apply Standard Scaling to numerical features.
   - Encode the target variable.
   - Fit preprocessing transformations on training data only to avoid data leakage.

4. **Model Building and Evaluation**
   - Train and evaluate multiple classification algorithms.
   - Compare training and testing performance.
   - Apply Stratified K-Fold Cross-Validation.
   - Evaluate cancellation-specific Precision, Recall, and F1-score.

5. **Hyperparameter Tuning**
   - Use `RandomizedSearchCV` to optimize the Random Forest model.
   - Use 5-fold Stratified Cross-Validation.
   - Optimize for the F1-score of the `Canceled` class.

6. **Model Saving and Prediction**
   - Save the selected fitted pipeline using Joblib.
   - Reload the saved model.
   - Verify that predictions remain consistent after reloading.
   - Implement reusable prediction functions.

7. **Streamlit Application**
   - Build a home page with project information and model performance.
   - Develop a booking prediction page.
   - Create a data insights page with interactive visualizations.

8. **Automated Testing**
   - Test preprocessing functionality.
   - Test model loading and prediction.
   - Validate prediction probabilities and output shapes.

## Model Development

The following classification algorithms were implemented and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Classifier (SVC)

Model evaluation included:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- Stratified K-Fold Cross-Validation

The initial Random Forest model achieved the strongest overall performance among the evaluated models. It was then selected for hyperparameter tuning.

### Final Model

**Tuned Random Forest Pipeline**

Hyperparameter optimization was performed using `RandomizedSearchCV` with:

- Number of parameter combinations: 30
- Cross-validation folds: 5
- Optimization metric: F1-score for the `Canceled` class
- Random state: 42

Best hyperparameters:

| Hyperparameter | Selected Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 20 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 1 |
| `max_features` | sqrt |
| `bootstrap` | False |

The final fitted pipeline includes categorical preprocessing and the Random Forest classifier. It accepts raw booking features and performs the required preprocessing internally.

## Final Model Performance

The final model was evaluated on the held-out test set.

### Test Set Results

| Metric | Score |
|---|---:|
| Accuracy | 89.31% |
| Canceled Precision | 87.43% |
| Canceled Recall | 78.68% |
| Canceled F1-score | 82.82% |

### Confusion Matrix

Class order: `Canceled`, `Not_Canceled`.

| Actual / Predicted | Canceled | Not_Canceled |
|---|---:|---:|
| Canceled | 1,871 | 507 |
| Not_Canceled | 269 | 4,610 |

The model correctly identified 1,871 canceled bookings and 4,610 bookings that were not canceled.

**Note:** The cancellation-specific metrics focus on the `Canceled` class because identifying potential cancellations is the main objective of this project.

## Project Structure

```text
hotel-booking-prediction/
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Booking_Prediction.py
│       └── 2_Data_Insights.py
│
├── data/
│   ├── raw/
│   │   └── booking.csv
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Data_Preprocessing.ipynb
│   └── Model_Building.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── predict.py
│   └── preprocessing.py
│
├── models/
│   └── best_random_forest_pipeline.joblib
│
├── tests/
│   ├── test_prediction.py
│   └── test_preprocessing.py
│
├── reports/
│   └── figures/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, model training, evaluation, and tuning |
| Joblib | Model serialization and loading |
| Streamlit | Interactive web application |
| Matplotlib & Seaborn | Data visualization |
| Jupyter Notebook | Exploratory analysis and model development |
| Pytest | Automated testing |
| Git & GitHub | Version control and project hosting |

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mahmoud1641998/Hotel-Booking-Cancellation-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd <YOUR_PROJECT_DIRECTORY>
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

From the project root directory, run:

```bash
streamlit run app/Home.py
```

Streamlit will start the application and provide a local URL, usually:

```text
http://localhost:8501
```

### Application Pages

**1. Home**

Provides an overview of the project, its objectives, workflow, and final model performance.

**2. Booking Prediction**

Allows users to enter booking information and receive:

- Predicted booking status.
- Cancellation probability.
- Non-cancellation probability.
- A visual representation of prediction probabilities.

**3. Data Insights**

Provides descriptive analysis and visualizations of the complete original booking dataset, including:

- Booking status distribution.
- Cancellation rate by market segment.
- Cancellation rate by room type.
- Average lead time by booking status.

The Data Insights page analyzes the original dataset, while model evaluation is performed separately using the held-out test set.

## Running Automated Tests

The project includes automated tests for preprocessing and prediction functionality.

Run the complete test suite from the project root:

```bash
python -m pytest -v
```

### Test Results

The complete test suite contains **12 tests**:

| Test Module | Number of Tests | Status |
|---|---:|---|
| `test_preprocessing.py` | 5 | Passed |
| `test_prediction.py` | 7 | Passed |
| **Total** | **12** | **All Passed** |

All 12 tests passed successfully in the local development environment.

## Future Improvements

Potential future improvements include:

- Deploying the application to Streamlit Community Cloud.
- Adding more interactive data exploration capabilities.
- Improving cancellation detection through further feature engineering and model experimentation.
- Monitoring model performance with new booking data.
- Expanding automated testing to cover additional application scenarios.

## Project Status

The Machine Learning workflow, final model, Streamlit application, and core automated tests have been completed and verified locally.

The project has been uploaded to GitHub. Cloud deployment and online application verification are the remaining delivery stages.

## Author

**Mahmoud Abdelnaby**

AI & Data Science Instructor | Machine Learning

[LinkedIn](https://www.linkedin.com/in/mahmoud-abdelnaby-6b53b9297/)

---

*This project was developed for practical learning and demonstration of an end-to-end Machine Learning workflow.*
