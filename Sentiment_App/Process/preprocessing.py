# preprocessing.py

import pandas as pd
import re
import nltk
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Mengunduh komponen yang diperlukan dari NLTK
nltk.download(['stopwords', 'punkt', 'wordnet'])  # Menggabungkan pengunduhan

# Fungsi untuk membersihkan teks
def cleaning_text(text):
    text = re.sub(r'https?://\S+|www\.\S+|@[\w]*', ' ', text)  # Memperbaiki regex
    text = re.sub(r'[^\w\s]', ' ', text)  # Menggunakan regex untuk menghapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
    return text

# Fungsi untuk menghapus stopwords
def remove_stopword(text, stop_words):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Fungsi utama untuk preprocessing data
def preprocess_data(df):
    df_cleaned = df.dropna(axis=1, how='all')
    df_cleaned['casefolding'] = df_cleaned['text'].str.lower()
    df_cleaned['cleanedtext'] = df_cleaned['casefolding'].apply(cleaning_text)

    # Menggabungkan stopwords dari NLTK dan custom stopwords
    sastrawi_stopword = "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/sastrawi-stopwords.txt"
    response = requests.get(sastrawi_stopword)
    stop_words = set(stopwords.words('indonesian') + response.text.split('\n'))
    custom_st = 'yg yang dgn ane smpai bgt gua gwa si tu ama utk udh btw nitar lol ttg emg aj aja tll sy sih kalo nya trsa mnrt nih'
    stop_words.update(custom_st.split())

    df_cleaned['stopwordremoved'] = df_cleaned['cleanedtext'].apply(lambda x: remove_stopword(x, stop_words))
    df_cleaned['lemmatizedtext'] = df_cleaned['stopwordremoved'].apply(lemmatize)
    df_cleaned['tokenize'] = df_cleaned['lemmatizedtext'].apply(word_tokenize)
    #df_cleaned.to_csv('otomotif-preprocessed.csv', index=False)
    return df_cleaned