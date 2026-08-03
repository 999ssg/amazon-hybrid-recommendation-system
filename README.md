# amazon-hybrid-recommendation-system
A hybrid recommendation system combining collaborative filtering (SVD) and content-based popularity models on Amazon product review data.

# 🛒 Amazon Product Recommendation System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/999ssg/amazon-hybrid-recommendation-system/blob/main/Amazon_Product_Recommendation_System_%E2%80%94Industry_Case_Study.ipynb)

A scalable, hybrid recommendation engine built on Amazon product interaction data, combining **Popularity-Based Filtering**, **Collaborative Filtering (Matrix Factorization / SVD)**, and **Item-Item Similarity** to solve the cold-start problem and deliver personalized product suggestions.

---

## 📌 Situation
E-commerce platforms face significant challenges with product discovery due to sparse user-item interaction matrices and the cold-start problem for new users. To increase user engagement and average order value (AOV), platforms require personalized recommendation pipelines capable of serving both historical users and new visitors.

---

## 🎯 Task
* Design and evaluate a **multi-tiered recommendation pipeline** using Amazon rating and review data.
* Mitigate matrix sparsity through data filtering and dimensionality reduction techniques.
* Implement a **Popularity-based Baseline Model**, **Collaborative Filtering (SVD/KNN)**, and a **Hybrid Model** that dynamically routes users based on interaction history depth.
* Evaluate models using cross-validation and standard metrics (**RMSE, MAE, Precision@K, Recall@K**).

---

## 🛠️ Action & Architecture
1. **Data Preprocessing & Matrix Pruning:**
   * Filtered explicit rating data to handle long-tail sparsity by setting interaction thresholds per user and product.
   * Built user-item interaction matrices using `pandas` and `scipy.sparse`.

2. **Model Engineering:**
   * **Popularity Model:** Calculated weighted ratings based on interaction counts to address cold-start users.
   * **Collaborative Filtering:** Implemented Matrix Factorization (SVD) and User/Item-based Neighborhood models using `Surprise` / `SciPy`.
   * **Hybrid Ensembling:** Created a fallback routing engine that serves collaborative recommendations to active users and switches to popularity metrics for cold-start users.

3. **Evaluation & Hyperparameter Tuning:**
   * Used GridSearch Cross-Validation to optimize latent factors ($k$), learning rate ($\gamma$), and regularization ($\lambda$) for SVD.

---

## 📈 Results & Impact
* **Error Reduction:** Achieved an optimized **RMSE of ~0.85–0.90** on explicit 1–5 scale ratings via SVD matrix factorization.
* **Cold-Start Handling:** Resolved zero-interaction failure modes by seamlessly routing non-historical users to weighted popularity lists.
* **Scalability:** Structured the prediction pipeline into clean, modular functions ready for batch inference or API integration.

---

## 📂 Repository Structure
