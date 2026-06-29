# 📈 Stock Price Prediction (AAPL)

## 📝 Overview
This project involves predicting the closing price of **Apple Inc. (AAPL)** stock using historical market data. It explores time-series analysis and regression modeling to understand market trends and make informed predictions.

## 🎯 Objective
The goal is to build a machine learning model that can forecast stock prices based on daily market features such as opening price, high/low values, and trading volume.

## 📂 File Description
*   `code.ipynb`: The comprehensive notebook covering data fetching, feature engineering, model training, and performance visualization.

## 🔄 Workflow
1.  **Data Loading**: Historical AAPL stock data (2020–2025) is downloaded directly from Yahoo Finance using the `yfinance` library.
2.  **Exploratory Data Analysis**: Inspecting the dataset's structure, missing values, and column interactions.
3.  **Feature Selection**: Selecting input variables (`Open`, `High`, `Low`, `Volume`) to predict the target variable (`Close`).
4.  **Train-Test Split**: Dividing the data into training (80%) and testing (20%) sets to ensure robust validation.
5.  **Model Training**: Implementing a **Linear Regression** model to map the relationship between features and the stock price.
6.  **Prediction**: Generating price forecasts on the test dataset.
7.  **Evaluation**: Measuring accuracy using **Mean Absolute Error (MAE)** to quantify prediction deviations.
8.  **Visualization**: Plotting actual vs. predicted prices to visually assess model performance.

## 🛠️ Techniques, Libraries, and Tools
*   **Data Fetching**: `yfinance`
*   **Data Processing**: `Pandas`, `NumPy`
*   **Visualization**: `Matplotlib`
*   **Machine Learning**: `Scikit-learn` (Linear Regression, Train-Test Split, MAE)

## 📊 Outputs / Results
*   **Mean Absolute Error**: ~0.74 (indicating high precision in price estimates).
*   **Visual Assessment**: The actual vs. predicted price plot shows a very high correlation, indicating that the model captures the overall trend effectively.

## 💡 Observations & Conclusions
*   Daily stock prices are heavily influenced by the same day's opening and high/low values, making Linear Regression a strong baseline for short-term estimation.
*   The model achieves impressive accuracy, though real-world stock trading would require further considerations like sentiment analysis and macro-economic factors.

---
*Created as part of Task 2.*
