# Car Price Prediction Using Machine Learning

## Table of Contents
1. [Overview](#overview)
2. [Project Features](#project-features)
3. [Data Summary](#data-summary)
4. [Preprocessing Steps](#preprocessing-steps)
5. [Visualizations and Analysis](#visualizations-and-analysis)
6. [Key Insights](#key-insights)
7. [How to Run the Project](#how-to-run-the-project)
8. [Results](#results)
9. [Future Improvements](#future-improvements)
10. [Acknowledgments](#acknowledgments)

## Overview
This project focuses on building an AI-based car price prediction model using various machine learning algorithms. The dataset comprises comprehensive details of over 19,000 cars, sourced from the "Car Price Prediction Challenge" on Kaggle. The goal is to leverage this data and train models that can accurately predict car prices based on features such as mileage, manufacturer, engine volume, and more.

## Project Features
- **Comprehensive Data Analysis**: Detailed Exploratory Data Analysis (EDA) including visualization of categorical and numerical features.
- **Data Preprocessing**: Cleaning data by handling missing values, transforming attributes, and standardizing numerical data.
- **Feature Engineering**: Creation of new features such as age calculation and handling of engine volume with turbo identifiers.
- **Outlier Removal**: Implemented outlier detection and removal using the Interquartile Range (IQR) method for improved model performance.
- **Model Building**: Used multiple machine learning models including:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Ridge Regression
  - XGBoost Regressor
  - MLP Regressor (Neural Network)
- **Performance Metrics**: Evaluated model performance using cross-validation and metrics like Root Mean Squared Error (RMSE).

## Data Summary
- **Dataset Size**: Initially 19,237 rows, reduced after preprocessing to 10,792 rows post-outlier removal.
- **Key Features**:
  - **Numerical**: Price, Levy, Engine volume, Mileage, Cylinders, Age, Airbags.
  - **Categorical**: Manufacturer, Model, Category, Fuel type, Drive wheels, Gearbox type, and more.

## Preprocessing Steps
1. **Missing Value Handling**: Used KNNImputer for imputing missing values.
2. **Feature Transformation**:
   - Converted production year to car age.
   - Extracted numeric engine volume and created a 'Turbo' feature.
   - Reformatted the 'Mileage' and 'Levy' columns for numerical consistency.
3. **Outlier Removal**:
   - Applied IQR filtering to remove outliers from columns like Price, Levy, Mileage, and Cylinders.

## Visualizations and Analysis
- **Histogram Analysis**: Visualized the distribution of features to identify skewness and data distribution patterns.
- **Correlation Heatmap**: Highlighted relationships between numerical attributes, identifying strong and weak correlations.
- **Scatter Plots**: Illustrated relationships between features such as engine volume and cylinders.

## Key Insights
- **Engine Volume and Cylinders**: Strong correlation indicating larger engine volumes have more cylinders.
- **Price Distribution**: Right-skewed, indicating most cars are priced in the lower range.
- **Feature Importance**: Assessed using feature correlation and importance plots from tree-based models.

## How to Run the Project
1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/car-price-prediction.git
  ```
2. Navigate to the project folder:

  ```bash
  cd car-price-prediction
  ```

3. Run the Jupyter Notebook:

```bash
jupyter notebook CarPricePrediction.ipynb
```

## Results
The project tested multiple algorithms, with XGBoost Regressor showing promising performance in terms of predictive accuracy. The overall analysis of model metrics indicated a successful reduction in RMSE after rigorous data preprocessing and model tuning.

## Future Improvements
- Hyperparameter Tuning: Use techniques like Grid Search for further optimization.
- Feature Expansion: Include additional data points such as regional economic factors that may impact car prices.
- Model Deployment: Create a web-based interface for real-time price prediction.

## Acknowledgments

- Kaggle for the dataset: [Car Price Prediction Challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge/data).
- Libraries used: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`.
