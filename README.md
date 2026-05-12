# 🤖 Intelligent Customer Support Chatbot using Machine Learning
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![CA2](https://img.shields.io/badge/Submission-CA2%20Project-purple)

---

## 📌 Project Overview

This project implements an **Intelligent Customer Support Chatbot** using **Machine Learning and Natural Language Processing (NLP)**.

It classifies customer queries into **8 intent categories** and responds with the most relevant support answer automatically — without any human agent involvement.

The system is built end-to-end: from raw text preprocessing, through TF-IDF feature extraction and multi-model classification, to cosine similarity-based response retrieval.

---

## 🎯 Objectives

✔ Automatically detect customer intent from support queries  
✔ Compare multiple ML algorithms for intent classification  
✔ Build a retrieval-based response engine using cosine similarity  
✔ Handle edge cases — greetings, farewells, thanks, and unknown queries  
✔ Connect the implementation to Transformer (BERT / Reformer) theory  
✔ Demonstrate a clean, modular, production-ready chatbot architecture  

---

## ✨ Features

✔ 8 Intent Categories covering all major e-commerce support scenarios  
✔ Five ML Models trained and compared (LR, SVM, RF, NB, GBoost)  
✔ TF-IDF based feature extraction (unigrams + bigrams)  
✔ Cosine Similarity response retrieval (intent-filtered corpus)  
✔ Confidence threshold fallback for unrecognised queries  
✔ Special handling for greetings, farewells, and thank-you messages  
✔ Conversation history tracking per session  
✔ Self-attention visualisation (Transformer concept demonstration)  
✔ Clean, modular class-based chatbot design  

---

## 🛠️ Tech Stack

### 🔹 Programming
* Python 3

### 🔹 Libraries
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

### 🔹 Machine Learning Models
* Logistic Regression
* Support Vector Machine (Linear Kernel)
* Random Forest
* Naive Bayes
* Gradient Boosting

### 🔹 NLP Techniques
* Text Preprocessing (lowercasing, punctuation removal, digit removal)
* TF-IDF Vectorisation (unigrams + bigrams)
* Cosine Similarity (response retrieval)
* Self-Attention (Transformer concept visualisation)

---

## 📂 Dataset

**File:** `customer_support_dataset.csv`

| Property | Value |
|---|---|
| Total Records | 40 |
| Columns | query, intent, response |
| Intent Classes | 8 |
| Samples per Intent | 5 (perfectly balanced) |
| Missing Values | None |
| Duplicate Records | None |

### 🔖 Intent Categories

| Intent Label | Description | Example Query |
|---|---|---|
| `order_tracking` | Track shipment and delivery status | *"Where is my package?"* |
| `returns_refunds` | Return items or get a refund | *"How do I get a refund?"* |
| `order_cancellation` | Cancel a placed order | *"Please cancel my purchase"* |
| `account_support` | Login, password, profile help | *"I forgot my password"* |
| `payment_issues` | Failed payments, double charges | *"My payment failed"* |
| `promotions` | Coupons, discounts, sales info | *"My coupon isn't working"* |
| `general_inquiry` | Shipping time, business hours | *"Do you ship internationally?"* |
| `product_complaint` | Wrong item, damage, missing parts | *"I received the wrong item"* |

**Labeling:**
* Each row has a human-written query, its intent label, and the ideal response
* The chatbot learns to map queries → intents → responses automatically

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading
Load `customer_support_dataset.csv` using Pandas and inspect shape, nulls, and class balance

### 2️⃣ Exploratory Data Analysis (EDA)
* Intent distribution (bar chart + pie chart)
* Query word count and character length distribution
* Average query length per intent
* Average response length per intent

### 3️⃣ Text Preprocessing
* Lowercasing all text
* Removing punctuation characters
* Removing digits and numbers
* Collapsing extra whitespace

### 4️⃣ Feature Extraction
TF-IDF Vectorisation with:
* `max_features = 500`
* `ngram_range = (1, 2)` — unigrams + bigrams
* `stop_words = 'english'`
* `sublinear_tf = True`

### 5️⃣ Model Training and Evaluation

| Model | Test Accuracy | Weighted F1 | CV Mean Accuracy |
|---|---|---|---|
| Logistic Regression | 50.00% | 0.4167 | 67.50% |
| **SVM (Linear)** | 50.00% | 0.3958 | **70.00%** ← Best CV |
| **Random Forest** | **62.50%** | **0.5208** | 57.50% |
| Naive Bayes | 50.00% | 0.4167 | 60.00% |
| **Gradient Boosting** | **62.50%** | **0.5208** | 62.50% |

> 📝 *Note: Low test scores are due to only 8 test samples (40 total). CV accuracy is the reliable metric. SVM achieves the best generalisation at 70% CV accuracy.*

### 6️⃣ Response Retrieval
Cosine similarity search within the intent-filtered training corpus returns the most contextually relevant response

### 7️⃣ Chatbot Integration
Full `CustomerSupportChatbot` class with special handling, confidence thresholding, and conversation history tracking

---

## 🧪 Manual Testing

```python
bot = CustomerSupportChatbot(df, tfidf, best_model, le)

# Test 1 — Standard support query
resp, intent, conf = bot.respond("Where is my order?")
# Intent     : order_tracking
# Confidence : 91%
# Response   : "Your package can be tracked using the tracking number sent to your email..."

# Test 2 — Greeting
resp, intent, conf = bot.respond("Hello!")
# Intent     : greeting
# Confidence : 100%
# Response   : "Hello! Welcome to Customer Support. I can help with order tracking..."

# Test 3 — Unknown query (fallback triggered)
resp, intent, conf = bot.respond("asdfgh xyzabc random text")
# Intent     : FALLBACK
# Confidence : <30%
# Response   : "I'm sorry, I couldn't understand that. Please try: track order | return item..."

# Test 4 — Thank you
resp, intent, conf = bot.respond("Thank you so much!")
# Intent     : thanks
# Confidence : 100%
# Response   : "You're welcome! Is there anything else I can help you with?"
```

---

## 💬 Sample Conversation

```
========================================================
      CUSTOMER SUPPORT CHATBOT — LIVE DEMO
========================================================

User   : Hello!
Intent : greeting  [██████████]  100%
Bot    : Hello! Welcome to Customer Support. I can help with order
         tracking, returns, cancellations, payments, and more.

User   : I want to return a damaged product
Intent : returns_refunds  [████████░░]  88%
Bot    : We sincerely apologize! Please take a photo of the damaged
         item and contact us with your order ID. We'll send a replacement.

User   : My payment was declined
Intent : payment_issues  [████████░░]  84%
Bot    : Payments can be declined due to insufficient funds or bank
         security checks. Please try again or use a different method.

User   : Thank you!
Intent : thanks  [██████████]  100%
Bot    : You're welcome! Is there anything else I can help you with?

User   : Bye!
Intent : farewell  [██████████]  100%
Bot    : Thank you for contacting us. Have a great day!
========================================================
```

---

## 🔬 Transformer Model — Course Alignment

This project connects to the course content on **Transformer Models** and the **Reformer**.

### Why Transformers Matter for Chatbots

| Aspect | TF-IDF + ML (Our System) | BERT (Transformer) |
|---|---|---|
| Word relationships | Words treated independently | Every word attends to every other word |
| Context awareness | Bag-of-words only | Full bidirectional context |
| Short query handling | Bigrams help partially | Self-attention captures full meaning |
| Complexity | O(n) | O(n²) |

### Unique Challenges Transformers Face

| Challenge | Description | Solution |
|---|---|---|
| O(n²) attention complexity | Self-attention is infeasible for long ticket threads | **Reformer** uses LSH attention → O(n log n) |
| Short query problem | 8-word queries give little context | Bigrams + similarity retrieval |
| Domain shift | Pre-trained BERT saw Wikipedia, not support tickets | Fine-tuning on the support corpus |
| Out-of-scope queries | Users ask unrecognised things | Confidence threshold triggers fallback |

### The Reformer Solution

The **Reformer** (Kitaev et al., 2020) uses **Locality-Sensitive Hashing (LSH) attention** — instead of comparing every word pair, it groups similar words into buckets and only computes attention within buckets. This brings complexity from **O(n²) → O(n log n)**, making it practical for full conversation threads.

---

## 🚀 How to Run

### 1️⃣ Install Python
Make sure Python 3.x is installed on your system

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3️⃣ Clone Repository

```bash
git clone https://github.com/your-repo-link/customer-support-chatbot.git
cd customer-support-chatbot
```

### 4️⃣ Place the Dataset
Make sure `customer_support_dataset.csv` is in the same folder as the notebook

### 5️⃣ Run the Notebook

```bash
jupyter notebook intelligent_customer_support_chatbot.ipynb
```

Run all cells from top to bottom. The chatbot will be ready to use at the end.

### 6️⃣ Start Interactive Chat (Optional)

Uncomment this line in the final cell of the notebook:

```python
interactive_chat()
```

---

## 📊 Project Outcomes

✔ Working end-to-end intelligent chatbot for e-commerce customer support  
✔ Five ML models trained, evaluated, and compared on real data  
✔ SVM achieved best cross-validation accuracy of 70%  
✔ Random Forest and Gradient Boosting led on test set metrics (62.5%)  
✔ Confidence threshold correctly activates fallback for unknown queries  
✔ Course alignment demonstrated through Transformer and Reformer analysis  
✔ Self-attention heatmap visualised for interpretability  

---

## 📁 Project Files

| File | Description |
|---|---|
| `customer_support_dataset.csv` | Dataset — 40 queries across 8 intent categories |
| `intelligent_customer_support_chatbot.ipynb` | Complete Jupyter notebook with all code |
| `Project_Report_CustomerSupportChatbot.pdf` | Full CA2 project report (10 sections) |
| `README.md` | This file |

---

## 🔮 Future Enhancements

🚀 Scale to full Bitext dataset (26,000+ samples)  
🚀 Fine-tune DistilBERT / BERT for 95%+ accuracy  
🚀 Add Sentence-BERT semantic retrieval  
🚀 Multi-turn context window for follow-up questions  
🚀 Web app deployment (Streamlit / Flask)  
🚀 REST API with FastAPI  
🚀 Docker containerisation  
🚀 Cloud deployment (AWS / GCP)  
🚀 Named Entity Recognition (extract order IDs, product names)  
🚀 Multilingual support using mBERT or XLM-R  

---

## 📚 References

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
2. Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL 2019.
3. Kitaev, N., et al. (2020). *Reformer: The Efficient Transformer*. ICLR 2020.
4. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT*. EMNLP 2019.
5. Bitext Customer Support Dataset — https://www.kaggle.com/datasets/bitext/
6. Customer Support on Twitter — https://www.kaggle.com/datasets/thoughtvector/
7. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12.

---

## 👨‍🎓 Authors

**[Team Lead Name]**  
**[Member 2 Name]**  
**[Member 3 Name]**  
Lovely Professional University

---

## 📚 Course

**CA2 Project — Natural Language Processing**

---

## 📅 Date

**2024 – 2025**

---

⭐ If you found this project useful, don't forget to star the repo!
