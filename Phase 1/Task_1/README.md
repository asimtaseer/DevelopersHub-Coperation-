# 🌸 Iris Dataset Exploratory Data Analysis (EDA)

## 📝 Overview
This task involves a comprehensive analysis of the classic **Iris Dataset** using Python. The project focuses on data exploration, statistical analysis, and advanced visualizations to understand the biological characteristics of three iris species.

## 🎯 Objective
The primary goal is to perform an in-depth Exploratory Data Analysis (EDA) to:
- Understand the distribution of sepal and petal measurements.
- Identify correlations between different physical attributes.
- Visualize the separability of species based on their features.

## 📂 File Description
- `code.ipynb`: The main Jupyter Notebook containing the Python code for data loading, processing, and visualization.
- `README.md`: Documentation providing an overview of the task.

## 🔄 Workflow & Process
The analysis follows a structured data science workflow:
1.  **Environment Setup**: Importing necessary libraries such as `Pandas`, `Seaborn`, and `Matplotlib`.
2.  **Data Loading**: Utilizing `seaborn` to import the built-in Iris dataset.
3.  **Basic Data Inspection**: 
    - Checking the dataset shape (`df.shape`).
    - Inspecting column types and missing values (`df.info()`).
    - Reviewing the first few records (`df.head()`).
4.  **Statistical Analysis**: Generating descriptive statistics (`df.describe()`) to understand the mean, standard deviation, and quartiles of the features.
5.  **Data Visualization**:
    - **Scatter Plots**: Visualizing the relationship between sepal length and width to observe species clustering.
    - **Histograms**: Analyzing the distribution and frequency of each feature.
    - **Box Plots**: Identifying outliers and comparing the distribution of features across different species.

## 🛠️ Techniques, Libraries, and Tools
- **Data Manipulation**: `Pandas`
- **Data Visualization**: `Matplotlib`, `Seaborn`
- **Computational Environment**: Jupyter Notebook

## 📊 Key Results
- **Species Separability**: Petal measurements show a clearer distinction between species compared to sepal measurements.
- **Statistical Summary**: The dataset consists of 150 samples with zero missing values, ensuring high data quality for analysis.
- **Visual Insights**: Species like *Setosa* exhibit unique feature distributions that make them easily identifiable.

## 💡 Important Observations
- The dataset is perfectly balanced with an equal number of samples for each species.
- Strong positive correlations exist between petal length and petal width.
- Petal features are the most significant indicators for species classification.

---
*Developed as part of Task 1.*
