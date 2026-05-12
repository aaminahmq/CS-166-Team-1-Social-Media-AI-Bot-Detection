# CS-166-Team-1-Social-Media-AI-Bot-Detection

Team 1 Members: Aaminah Mohammed, Alex Chang, Jumana Ayoub

Data Source: kaggle.com/datasets/danieltreiman/twitter-human-bots-dataset 

# Social Media Bot Detection Using Machine Learning

**CS166 Information Security | Team 1 | Spring 2026**  
San Jose State University | Instructor: Professor Chao-Li Tarng

---

## Team Members

- Jumana Ayoub
- Aaminah Mohammed
- Alex Chang

---

## Project Overview

This project builds a machine learning pipeline to automatically detect bot accounts on social media platforms. We trained and compared five supervised classifiers on a real labeled Twitter dataset containing 37,438 accounts. The best performing model, Random Forest, achieved 88.22% accuracy and an AUC-ROC of 0.9348.

The project also includes a live web dashboard built with Streamlit that allows users to enter account details and receive an instant bot or human prediction.

---

## Repository Structure

```
CS-166-Team-1-Social-Media-AI-Bot-Detection/
│
├── app.py                          # Streamlit live detector dashboard
├── CS166_Bot_Detection.ipynb       # Full ML pipeline notebook
├── README.md                       # This file
```

---

## Dataset

We used a real labeled Twitter dataset containing 37,438 accounts (25,013 human, 12,425 bot).

The dataset is loaded directly from GitHub in both the notebook and the dashboard, so no manual download is required.

link to dataset: kaggle.com/datasets/danieltreiman/twitter-human-bots-dataset
```

---

## Features Used

The following 12 features were selected from the account metadata:

| Feature | Description |
|---|---|
| `followers_count` | Number of followers |
| `friends_count` | Number of accounts following |
| `favourites_count` | Total likes given |
| `statuses_count` | Total tweets posted |
| `verified` | Whether the account is verified |
| `default_profile` | Whether the account uses default profile settings |
| `default_profile_image` | Whether the account uses the default profile image |
| `geo_enabled` | Whether geo location is enabled |
| `average_tweets_per_day` | Average number of tweets posted per day |
| `account_age_days` | Age of the account in days |
| `description_length` | Number of characters in the bio |
| `followers_friends_ratio` | Ratio of followers to accounts following |

---

## Models Compared

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest | **0.8822** | 0.8573 | 0.7686 | **0.8106** | **0.9348** |
| XGBoost | 0.8790 | 0.8447 | 0.7731 | 0.8073 | 0.9336 |
| Decision Tree | 0.8562 | 0.8051 | 0.7405 | 0.7715 | 0.8808 |
| KNN (K=5) | 0.8002 | 0.6925 | 0.7026 | 0.6975 | 0.8487 |
| Logistic Regression | 0.7615 | 0.6459 | 0.6033 | 0.6238 | 0.8222 |

---

## Running the Live Dashboard

### Requirements

Make sure you have Python 3.8 or higher installed. Then install the required libraries by running:

```bash
pip install streamlit scikit-learn pandas numpy xgboost
```

### Steps to Run

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/CS-166-Team-1-Social-Media-AI-Bot-Detection.git
cd CS-166-Team-1-Social-Media-AI-Bot-Detection
```

**2. Run the Streamlit app**

```bash
streamlit run app.py
```

If the above command does not work, try:

```bash
python -m streamlit run app.py
```

**3. Open in your browser**

Streamlit will automatically open the app at:

```
http://localhost:8501
```

The app will load and train the model on first launch. This takes about 30 to 60 seconds depending on your internet connection and machine.

---

## Using the Dashboard

Once the app is running:

1. Enter the account details in the input fields on the left side of the form
2. Click **Run Detection**
3. The result will appear below showing either **Bot Account Detected** or **Human Account** along with a confidence percentage
4. A signal breakdown table shows which features were flagged as suspicious

### Sample Accounts for Testing

**Bot Account (expected result: Bot)**

| Field | Value |
|---|---|
| Followers | 8 |
| Following | 4500 |
| Total Likes Given | 0 |
| Total Tweets Posted | 48000 |
| Avg Tweets / Day | 180.0 |
| Account Age (days) | 6 |
| Bio Length | 0 |
| Verified | No |
| Default Profile | Yes |
| Default Profile Image | Yes |
| Geo Enabled | No |

**Human Account (expected result: Human)**

| Field | Value |
|---|---|
| Followers | 843 |
| Following | 290 |
| Total Likes Given | 4200 |
| Total Tweets Posted | 1650 |
| Avg Tweets / Day | 2.5 |
| Account Age (days) | 1380 |
| Bio Length | 92 |
| Verified | No |
| Default Profile | No |
| Default Profile Image | No |
| Geo Enabled | Yes |

---

## Running the Notebook

The full ML pipeline is in `CS166_Bot_Detection.ipynb`. It covers:

- Data loading and exploration
- Missing value handling
- Feature engineering
- Train / test split and scaling
- Training and evaluating all 5 models
- Model comparison table, ROC curves, and feature importance charts

To run it locally, open it in Jupyter Notebook or upload it to Google Colab and run all cells from top to bottom.

