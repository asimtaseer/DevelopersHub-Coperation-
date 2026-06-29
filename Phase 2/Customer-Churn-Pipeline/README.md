---
title: Customer Churn Prediction System
emoji: 🔮
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
short_description: Predict telecom customer churn using a trained ML Pipeline.
tags:
  - machine-learning
  - classification
  - scikit-learn
  - telecom
  - churn-prediction
---

# 🔮 Customer Churn Prediction System

> Predict whether a telecom customer is likely to churn using a trained Scikit-Learn Machine Learning Pipeline.

---

## 🚀 Live Demo

Try the live app directly on [Hugging Face Spaces](https://huggingface.co/spaces/asimtaseer/customer-churn-pipeline).

---

## 📌 Overview

This application uses a trained **Logistic Regression** pipeline (part of a full Scikit-Learn preprocessing + modeling pipeline) to predict customer churn in the telecom industry. It is deployed as an interactive **Gradio** web application with a modern, professional SaaS-style UI.

| Metric        | Value         |
|---------------|---------------|
| Algorithm     | Logistic Regression |
| Accuracy      | **80.55 %**   |
| ROC-AUC       | **84.18 %**   |
| Framework     | Scikit-Learn Pipeline |
| Deployment    | Hugging Face Spaces (Gradio) |

---

## 🧩 Features

- **3 intuitive input sections** — Customer Info, Telecom Services, Billing Details
- **Instant predictions** with confidence scores and animated progress bars
- **Color-coded result cards** — Red (Churn) / Green (Safe)
- **Google Cloud-style UI** — clean cards, soft shadows, Inter typography
- **Mobile responsive** layout
- **Graceful error handling** with user-friendly messages

---

## 🗂️ Input Features

The model expects exactly **19 features** in this order:

| # | Column | Type |
|---|--------|------|
| 1 | `gender` | Categorical |
| 2 | `SeniorCitizen` | Integer (0/1) |
| 3 | `Partner` | Categorical |
| 4 | `Dependents` | Categorical |
| 5 | `tenure` | Integer |
| 6 | `PhoneService` | Categorical |
| 7 | `MultipleLines` | Categorical |
| 8 | `InternetService` | Categorical |
| 9 | `OnlineSecurity` | Categorical |
| 10 | `OnlineBackup` | Categorical |
| 11 | `DeviceProtection` | Categorical |
| 12 | `TechSupport` | Categorical |
| 13 | `StreamingTV` | Categorical |
| 14 | `StreamingMovies` | Categorical |
| 15 | `Contract` | Categorical |
| 16 | `PaperlessBilling` | Categorical |
| 17 | `PaymentMethod` | Categorical |
| 18 | `MonthlyCharges` | Float |
| 19 | `TotalCharges` | Float |

---

## 🛠️ Tech Stack

- **Frontend / App**: [Gradio](https://gradio.app/) ≥ 4.44
- **ML Framework**: [Scikit-Learn](https://scikit-learn.org/) ≥ 1.3
- **Model Hub**: [Hugging Face Hub](https://huggingface.co/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)

---

## 📦 Model Loading

The trained pipeline is loaded directly from Hugging Face Hub at runtime:

```python
from huggingface_hub import hf_hub_download
import joblib

model = joblib.load(
    hf_hub_download(
        repo_id="asimtaseer/customer-churn-pipeline",
        filename="sklearn_model.joblib"
    )
)
```

---

## 🏃 Run Locally

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/asimtaseer/customer-churn-pipeline
cd customer-churn-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
python app.py
```

The app will be available at `http://127.0.0.1:7860`.

---

## 📁 Project Structure

```
customer-churn-pipeline/
│
├── app.py              ← Gradio application (UI + prediction logic)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 👤 Author

**Asim Qurashi** — AI Engineer

- 🐙 [GitHub](https://github.com/asimtaseer)
- 💼 [LinkedIn](https://linkedin.com/in/asimtaseer)
- 🤗 [Hugging Face](https://huggingface.co/asimtaseer)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
