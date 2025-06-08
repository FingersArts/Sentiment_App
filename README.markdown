# Sentiment Analysis App (Automotive Topics)

This repository contains a sentiment analysis application that classifies text related to automotive topics (e.g., car reviews, forum discussions) into positive, negative, or neutral sentiments. The project uses **Multinomial Logistic Regression** as the core machine learning model, implemented in Python, to analyze automotive-themed datasets.

## Features
- **Sentiment Classification**: Predicts sentiment (positive, negative, neutral) for automotive-related text inputs.
- **Multinomial Logistic Regression**: Trained model for accurate multi-class sentiment prediction.
- **Automotive Focus**: Tailored for analyzing car reviews, customer feedback, or automotive forum posts.
- **Data Preprocessing**: Includes text cleaning, tokenization, stemming, and feature extraction (TF-IDF).

## Tech Stack
- **Programming Language**: Python
- **Machine Learning**: Multinomial Logistic Regression
- **Data Processing**: pandas, NumPy
- **Dependencies**: Listed in `requirements.txt`

## Dataset
The project uses a dataset of automotive-related text, such as car reviews or forum discussions, labeled with sentiments (positive, negative, neutral). The data is preprocessed to remove noise, apply tokenization, and convert text into numerical features using TF-IDF vectorization.


## Model Details
- **Algorithm**: Multinomial Logistic Regression
- **Features**: TF-IDF vectors derived from preprocessed text
- **Training**: Model trained on labeled automotive text data
- **Evaluation**: Metrics like accuracy, precision, recall, and F1-score

---

Developed by FingersArts 🚗
