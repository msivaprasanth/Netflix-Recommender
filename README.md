# 🎬 Netflix Recommender

A **hybrid movie recommendation system** that combines **Apriori association rules**, **NLP-based similarity (TF-IDF + cosine similarity)**, and a **Flask web application** — all styled with a **Netflix-inspired UI**.

---

## 📊 Dataset

**Source:** [Top 10000 Popular Movies - Kaggle](https://www.kaggle.com/datasets/omkarborikar/top-10000-popular-movies/data)

| Column Name | Description |
|--------------|-------------|
| `id` | Unique ID for each movie |
| `original_language` | Language code (ISO 639-1), e.g., `'en'` for English, `'hi'` for Hindi |
| `original_title` | Movie title |
| `popularity` | Popularity score (higher = more popular) |
| `release_date` | Date of release (if missing, movie not released yet) |
| `vote_average` | Average user rating |
| `vote_count` | Total number of ratings |
| `genre` | List of genres associated with the movie |
| `overview` | Brief movie description |
| `revenue` | Total box office revenue |
| `runtime` | Movie duration in minutes |
| `tagline` | Tagline of the movie |

---

## 🧠 Features

✅ **Hybrid Recommendation Approach**  
- **Language-based filtering** (e.g., Hindi movies → only Hindi recommendations)  
- **Genre association mining** using Apriori algorithm  
- **Semantic similarity** between `overview` + `tagline` using **TF-IDF + cosine similarity**  
- **Weighted priority** → Language → Genre → Overview+Tagline → Ratings  

✅ **Netflix-Inspired Frontend**  
- Dark, elegant UI with **red-accented theme**  
- Google-style **search bar** for movie input  
- Responsive & minimal design (HTML + CSS + Flask templates)

✅ **Explainable Recommendations**  
Each recommendation is chosen based on shared:
- Language  
- Genre overlap  
- Storyline similarity  
- Audience ratings  

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Backend | Python, Flask |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn, mlxtend |
| NLP | TF-IDF Vectorizer, Cosine Similarity |
| Visualization (optional) | NetworkX, Seaborn |
| Frontend | HTML5, CSS3 (Netflix Theme) |

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/msivaprasanth/Netflix-Recommender.git
cd Netflix-Recommender
