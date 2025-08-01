# 🍽️ Zomato Cuisine Clustering using VADER & K-Means

This notebook performs unsupervised clustering of Zomato restaurant reviews using:
- **TF-IDF** vectorization of review text
- **VADER sentiment analysis** for polarity scoring
- **Ratings** as numeric input
- **K-Means clustering** to uncover patterns in customer preferences

### 🔍 Goal:
To group reviews into meaningful clusters that can be used for **restaurant/cuisine recommendation systems**.

### 🛠 Tools Used:
- `pandas`, `scikit-learn`, `nltk` (VADER)
- `TfidfVectorizer` for text embedding
- `KMeans` for clustering

### ✅ Features Extracted:
- TF-IDF of `review` text  
- Sentiment scores (`compound`, `pos`, `neu`, `neg`) from VADER  
- Normalized `rating` score  

### 🎯 Output:
- Cluster-labeled reviews
- A function to recommend similar reviews based on new input

---

