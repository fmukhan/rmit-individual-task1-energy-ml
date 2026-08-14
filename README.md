# RMIT Individual Task 1 – Energy Consumption Prediction

This repository contains the machine learning analysis developed for **Individual Task 1: Part 1** in *Case Studies in Data Science* at RMIT University. The project applies two regression algorithms to two publicly available energy-consumption datasets and compares their predictive performance.

## Project Overview

The analysis uses:

- **Linear Support Vector Regression (Linear SVR)**
- **MLP Neural Network (MLPRegressor)**

Both datasets are analysed separately using a **chronological 80/20 train-test split**. Predictor variables are standardised before modelling, and model performance is evaluated using **MAE, RMSE, and R²**. Permutation importance is also used to identify influential predictors.

## Repository Files

| File | Description |
|---|---|
| `Appliances_Energy(1).ipynb` | Google Colab notebook for household appliance-energy prediction |
| `Tetouan_Power(1).ipynb` | Google Colab notebook for Tetouan Zone 1 electricity-consumption prediction |
| `energydata_complete(2).csv` | Appliances Energy Prediction dataset |
| `Tetuan City power consumption.csv` | Tetouan City Power Consumption dataset |

## Datasets

### 1. Appliances Energy Prediction

- **Size:** 19,735 rows × 29 columns
- **Target:** `Appliances`
- **Key attributes:** appliance energy use, lighting, indoor temperature and humidity measurements, outdoor weather variables, and time information.
- The notebook creates additional cyclical time features and excludes the documented random variables `rv1` and `rv2` from the predictive feature set.

Source: UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

### 2. Tetouan City Power Consumption

- **Size:** 52,416 rows × 9 columns
- **Target:** `Zone 1 Power Consumption`
- **Key attributes:** temperature, humidity, wind speed, general diffuse flows, diffuse flows, and electricity consumption for three city zones.
- The notebook creates cyclical time features and excludes Zone 2 and Zone 3 consumption from the predictors when modelling Zone 1.

Source: UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city

## Model Evaluation

The notebooks report:

- **MAE (Mean Absolute Error):** average absolute prediction error
- **RMSE (Root Mean Squared Error):** penalises larger prediction errors more strongly
- **R² (Coefficient of Determination):** indicates how much variation in the target is explained by the model

### Recorded Results

#### Appliances Energy

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear SVR | 47.3186 | 84.6618 | 0.1352 |
| MLP Neural Network | 141.9905 | 198.3314 | -3.7461 |

For this dataset, **Linear SVR performed better**. Permutation importance identified variables such as `RH_2`, `RH_1`, and `T2` among the strongest predictors.

#### Tetouan Zone 1

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| MLP Neural Network | 3454.4243 | 4278.2560 | 0.6440 |
| Linear SVR | 3495.8729 | 4289.5619 | 0.6421 |

For this dataset, the **MLP Neural Network performed slightly better**. Time-of-day features, general diffuse flows, and temperature were among the influential predictors.

## How to Run the Notebooks

The notebooks are configured for **Google Colab with Google Drive**.

1. Open the required `.ipynb` notebook in Google Colab.
2. Upload the corresponding CSV file to your Google Drive.
3. For the Appliances notebook, either:
   - rename `energydata_complete(2).csv` to `energydata_complete.csv`, **or**
   - update the `DATA_PATH` variable in the notebook.
4. For the Tetouan notebook, the expected Drive filename is:
   - `Tetuan City power consumption.csv`
5. Run the notebook cells from top to bottom.
6. The notebooks save model-comparison tables, predictions, permutation-importance results, and graphs to Google Drive.

## Main Python Libraries

- pandas
- numpy
- matplotlib
- scikit-learn

The notebooks use `LinearSVR`, `MLPRegressor`, `StandardScaler`, `TransformedTargetRegressor`, regression evaluation metrics, and permutation importance from scikit-learn.

## Reproducibility

A fixed `random_state=42` is used in the machine learning models and permutation-importance analysis where applicable. The chronological split is used instead of a random split so that earlier observations are used for training and later observations for testing.

---

**RMIT University – Individual Task 1: Part 1**
