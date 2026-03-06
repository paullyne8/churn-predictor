# 📡 AI-Driven Telecom Customer Churn Prediction System

> **Final Year Dissertation Project** — BSc (Hons) Computer Science, Middlesex University Mauritius (2026)

An end-to-end machine learning system for predicting customer churn in the telecommunications industry, combining **XGBoost**, **SHAP-based explainability**, and **LLM-powered natural language insights** — delivered as a production-ready **Streamlit web application**.

---

## 🧠 Project Overview

Customer churn is a critical business challenge in the telecom sector. This system goes beyond standard churn prediction by making AI decisions **transparent and explainable** — generating plain-English summaries of why a customer is at risk, powered by the OpenAI API.

### Key Features

- **Churn Prediction** — XGBoost classifier trained on telecom customer data with full evaluation (accuracy, precision, recall, F1, AUC-ROC)
- **Explainable AI** — SHAP values for both global feature importance and individual prediction explanations
- **LLM Integration** — OpenAI GPT generates plain-English summaries of SHAP outputs for non-technical stakeholders
- **Natural Language Interface** — users can query the model and its predictions conversationally
- **Interactive Dashboard** — full Streamlit web app with visualisations, predictions, and explanations

---

## 🏗️ Project Structure

```
churn-predictor/
│
├── app/
│   └── streamlit_app.py        # Main Streamlit web application
│
├── data/
│   ├── raw/                    # Raw dataset (not tracked in git)
│   └── processed/              # Cleaned and feature-engineered data
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # EDA and data exploration
│
├── src/
│   ├── preprocess.py           # Data cleaning and feature engineering
│   ├── train.py                # XGBoost model training and evaluation
│   ├── explain.py              # SHAP explainability module
│   └── llm_insights.py         # OpenAI API integration
│
├── models/
│   └── xgboost_model.pkl       # Saved trained model (not tracked in git)
│
├── outputs/
│   └── shap_plots/             # Generated SHAP visualisations
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost |
| Explainability | SHAP |
| LLM / NLP | OpenAI API (GPT) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Web Application | Streamlit |
| Visualisation | Matplotlib, Seaborn, SHAP plots |
| Language | Python 3.10+ |

---

## 📊 Dataset

This project uses the **IBM Watson Telco Customer Churn Dataset**, a widely used benchmark dataset in churn prediction research containing ~7,000 telecom customers with 21 features including:

- Customer demographics (gender, senior citizen status, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, streaming, security)
- Target variable: `Churn` (Yes/No)

---

## 🚀 Getting Started

### Prerequisites

```bash
python 3.10+
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor

# Install dependencies
pip install -r requirements.txt
```

### Set up environment variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

> ⚠️ Never commit your API key. The `.env` file is listed in `.gitignore`.

### Run the application

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Model Performance

> *Results will be updated as the project progresses.*

| Metric | Score |
|---|---|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |
| AUC-ROC | TBD |

---

## 🔍 Explainability Approach

This project uses **SHAP (SHapley Additive exPlanations)** to provide two levels of explanation:

- **Global explanations** — which features most influence churn predictions across all customers
- **Local explanations** — why the model predicted churn (or not) for a specific individual customer

SHAP outputs are then passed to an **OpenAI GPT model** which generates a plain-English summary suitable for business stakeholders, removing the need to interpret raw model outputs manually.

---

## 💬 LLM Integration

The OpenAI API is used for two purposes:

1. **Explanation summarisation** — converting SHAP feature importance into readable business insights
2. **Conversational interface** — allowing users to ask natural language questions about predictions and model behaviour

---

## 📚 References

- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*
- Lundberg, S. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*
- IBM Watson Telco Customer Churn Dataset — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 👩‍💻 Author

**Pauline Mtallo**
BSc Computer Science — Middlesex University, Mauritius
[LinkedIn](https://www.linkedin.com/in/paulinemtallo) · paulinemtallo@gmail.com

---

*This project is submitted in partial fulfilment of the requirements for the BSc (Hons) Computer Science degree at Middlesex University.*
