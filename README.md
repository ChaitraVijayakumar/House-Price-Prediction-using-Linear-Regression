# House-Price-Prediction using Linear-Regression & Decision-Tree
## Problem Statement
Predict the price of a house based on various features like area, number of bedrooms, bathrooms, and other amenities using *Linear Regression* & *Decision Tree*.

## Dataset
- Source: Kaggle Housing Prices Dataset
- Total Samples: 545
- Target Variable: `price` (House Price)
- Key Features: Area, Bedrooms, Bathrooms, Stories, Parking, Air Conditioning, etc.


## Technologies Used
- Python 3
- Pandas & NumPy (Data Handling)
- Matplotlib & Seaborn (Visualization)
- Scikit-Learn (Model Building)
- Jupyter Notebook

## Workflow
1. Data Loading & Exploration
2. Exploratory Data Analysis (EDA) with visualizations
3. Data Preprocessing (Label Encoding for categorical features)
4. Feature Selection
5. Train-Test Split (80-20)
6. Model Training using Linear Regression
7. Model Evaluation (MSE & R² Score)
8. Save the trained model

## Results
- **R² Score**: [Will be visible after running notebook]
- **Mean Squared Error**: [Will be visible after running notebook]

**Key Insights**:
- Larger area strongly increases house price
- Air conditioning, parking, and number of stories significantly affect price
- Furnishing status also plays an important role

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/House-Price-Prediction.git

2. Install required packages:
Bash
pip install -r requirements.txt

3. Open Jupyter Notebook:
Bash
jupyter notebook

4. Run housePrices.ipynb

## requirements.txt :
pandas\n
numpy
matplotlib
seaborn
scikit-learn
jupyter/vs code

## COMPARING LINEAR REGRESION & DECISION TREE BASED ON PERFORMANCE ##

## Model Comparison:

| Model                | R² Score     | MSE                     | Remarks                           |
|----------------------|--------------|-------------------------|-----------------------------------|
| Linear Regression    | [0.6495]     | [1,771,751,116,597.03]  | Simple & Interpretable            |
| Decision Tree        | [0.4585]     | [2,737,130,474,262.15]  | Better at capturing non-linearity |

**Observation**: 
Linear regression excels when features share a direct, proportional relationship with the target variable, making it ideal for straightforward numeric forecasting. Conversely, a decision tree performs better when dealing with complex, non-linear relationships, multi-level feature interactions, or a mix of numeric and categorical data.