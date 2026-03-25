# 🚀 AI-Powered E-Commerce Ads Optimization Engine

An end-to-end **Machine Learning + RAG-based system** that helps optimize e-commerce ads by generating:

* 🧠 AI-powered product titles
* 🔑 High-performing keywords
* ⚔️ Competition analysis
* 💰 Suggested bidding strategy

Built using **ML, NLP, LLMs (Groq), and MLOps (CI/CD + Docker)**.

---

## 📌 Problem Statement

In e-commerce advertising (Amazon, Flipkart, Q-commerce), selecting:

* the right **keywords**
* optimal **bids**
* high-converting **titles**

is critical but complex.

This project automates that process using:

* Real scraped marketplace data
* Machine Learning
* Retrieval-Augmented Generation (RAG)

---

## 🧠 System Architecture

```
User Input (Category + Brand + Keywords)
        ↓
Keyword Engine (data-driven)
        ↓
Competition + Bid Prediction (ML)
        ↓
RAG Retrieval (similar titles)
        ↓
LLM (Groq) → Title Generation
        ↓
Final Output
```

---

## ⚙️ Features

### 🔑 Keyword Suggestion

* Extracts top keywords from real product data
* Category-specific filtering
* Title-based keyword mining

### 🧠 AI Title Generation (RAG + LLM)

* Retrieves similar product titles (FAISS)
* Uses **Groq LLaMA3 model**
* Generates SEO-optimized product titles

### 📊 Competition Analysis

* Computes competition score (0–1)
* Classifies into:

  * LOW
  * MEDIUM
  * HIGH

### 💰 Bid Prediction (ML)

* Models used:

  * Random Forest
  * XGBoost
  * Linear Regression
* Best model selected via MLflow

---

## 🛠️ Tech Stack

| Category   | Tools                          |
| ---------- | ------------------------------ |
| ML         | Scikit-learn, XGBoost          |
| NLP        | Pandas, Text Processing        |
| LLM        | Groq (LLaMA3)                  |
| RAG        | FAISS + HuggingFace Embeddings |
| MLOps      | MLflow, CI/CD                  |
| Backend    | Python                         |
| UI         | Streamlit                      |
| Deployment | Docker                         |

---




## 🚀 Setup Instructions

### 1️⃣ Clone Repo

```bash
git clone https://github.com/Harsh6063/ecommerce-ad-ml_title_generator_keyword_finder.git
cd ecommerce-ad-ml
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Add Environment Variables

Create `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 4️⃣ Run Pipeline

```bash
python amazon_scraper.py 
python preprocess.py
python train_model.py
```

---

### 5️⃣ Run App

```bash
streamlit run app.py
```

---

## 🧪 CI/CD

* GitHub Actions pipeline:

  * installs dependencies
  * runs preprocessing
  * trains model
  * validates app

---

## 🐳 Docker Support

```bash
docker build -t ad-ml-app .
docker run -p 8501:8501 ad-ml-app
```

---

## 📊 Example Output

**Input:**

```
Category: Beauty
Brand: Lakme
Keyword1: skin
Keyword2: glow
```

**Output:**

```
Title: Lakme Skin Glow Face Cream with Vitamin C Brightening Formula
Keywords: skin, glow, cream, face, serum
Competition: HIGH
Score: 0.82
Bid: ₹18.5
```

---

## ⚠️ Notes

* Models and processed data are **not stored in repo**
* They are generated during runtime / CI
* MLflow tracking is enabled locally

---



Give it a ⭐ on GitHub and connect!

---
