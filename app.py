import streamlit as st
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from preprocessing import preprocess_and_calculate_tfidf, LogisticRegression
from preprocessing import plot_confusion_matrix, k_fold_cross_validation
import matplotlib.pyplot as plt

# Title of the application
st.title('Multinomial Logistic Regression Stage')

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Upload Data", "Preprocessing", "TF-IDF Results", "Multinomial Logistic Regression"])

# Tab for uploading data
with tab1:
    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # If a file is uploaded
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the original DataFrame
        st.subheader('Original Data')
        st.write(df)

# Tab for pre-processing
with tab2:
    if uploaded_file is not None:
        # Button to start preprocessing
        if st.button('Start Preprocessing'):
            # Process the dataframe
            preprocessed_df, ranking = preprocess_and_calculate_tfidf(df)
            
            st.subheader('Data After Preprocessing')
            st.write(preprocessed_df[['text', 'casefolding', 'cleanedtext', 'stopwordremoved', 'stemmedtext', 'tokenize']])

            # Count word frequencies in the 'stemmedtext' column
            word_counts = preprocessed_df['stemmedtext'].str.split(expand=True).stack().value_counts()
            
            # Visualize the top 10 most frequent words
            st.subheader('Preprocessed Data Visualization')
            st.bar_chart(word_counts.head(10))

            # Display the top 10 most frequent words
            st.subheader('Top 10 Words')
            st.write(word_counts.head(10))
    else:
        st.write("Please upload a CSV file in the 'Upload Data' tab to get started.")

# Tab for TF-IDF results
with tab3:
    if uploaded_file is not None:
        if 'preprocessed_df' in locals():
            # Display all TF-IDF ranked terms
            st.subheader('All Terms by TF-IDF')
            st.dataframe(ranking)
        else:
            st.write("Please go to the 'Preprocessing' tab and start the Preprocessing first.")
    else:
        st.write("Please upload a CSV file in the 'Upload Data' tab to get started.")

# Tab for Logistic Regression
with tab4:
    if uploaded_file is not None:
        if 'preprocessed_df' in locals():
            # Map sentiment_val to label
            sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
            df['label'] = df['sentiment_val'].map(sentiment_mapping)

            # Prepare data for Logistic Regression
            words = ' '.join(preprocessed_df['stemmedtext']).split()
            #words = ['some_number' if word.isnumeric() else word for word in words]
            #stop_words = set(stopwords.words('indonesian'))
            #stop_words.update(['-', '*', '&', '.', '/', ':', '|'])
            #words = [word for word in words if word not in stop_words]
            word_counts = Counter(words)
            features = [word for word, count in word_counts.most_common(1000)]

            def transform_to_features(texts, features):
                def count_features(words):
                    return [words.count(feature) for feature in features]
                return np.array([count_features(text.split()) for text in texts])

            X = transform_to_features(preprocessed_df['stemmedtext'], features)
            y = df['label'].values.astype(int)  # Use the newly created label column

            # Initialize and train the model with default parameters
            model = LogisticRegression(num_iter=200, learning_rate=0.44)
            (mean_accuracy, mean_precision, mean_recall, average_confusion, avg_class_accuracies, avg_precision_per_class, avg_recall_per_class) = k_fold_cross_validation(X, y, model, k=10)

            # Display model performance
            st.subheader('Model Performance')

            st.markdown("### Average Class Accuracies:")
            st.markdown(f"**Sentiment negative**: {avg_class_accuracies[0]:.6f}")
            st.markdown(f"**Sentiment neutral**: {avg_class_accuracies[1]:.6f}")
            st.markdown(f"**Sentiment positive**: {avg_class_accuracies[2]:.6f}")

            st.markdown("### Average Precision per Class:")
            st.markdown(f"**Sentiment negative**: {avg_precision_per_class[0]:.6f}")
            st.markdown(f"**Sentiment neutral**: {avg_precision_per_class[1]:.6f}")
            st.markdown(f"**Sentiment positive**: {avg_precision_per_class[2]:.6f}")

            st.markdown("### Average Recall per Class:")
            st.markdown(f"**Sentiment negative**: {avg_recall_per_class[0]:.6f}")
            st.markdown(f"**Sentiment neutral**: {avg_recall_per_class[1]:.6f}")
            st.markdown(f"**Sentiment positive**: {avg_recall_per_class[2]:.6f}")

            st.subheader('Confusion Matrix')

            confusion_df = pd.DataFrame(average_confusion, 
                            index=['True Negative', 'True Neutral', 'True Positive'], 
                            columns=['Predicted Negative', 'Predicted Neutral', 'Predicted Positive'])

            st.table(confusion_df)

            plot_confusion_matrix(average_confusion)
            st.pyplot(plt)

            st.subheader('Analisis Sentimen selama 1 Minggu Terakhir')

            # Konversi kolom 'date' ke format datetime dan tambahkan kolom 'tanggal'
            df['date'] = pd.to_datetime(df['date'])
            df['tanggal'] = df['date'].dt.strftime('%Y-%m-%d')

            # Filter data untuk 1 minggu terakhir
            df_last_week = df[df['date'] >= df['date'].max() - pd.to_timedelta(7, unit='D')]

            # Hitung jumlah sentimen untuk setiap tanggal dengan label 0 = negative, 1 = netral, dan 2 = positive
            sentiment_counts = df_last_week.groupby(['tanggal', 'label']).size().unstack(fill_value=0)

            # Hitung persentase untuk setiap sentimen
            sentiment_percentages = sentiment_counts.apply(lambda x: x/x.sum(), axis=1)

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(sentiment_percentages.index, sentiment_percentages[0], label='Sentimen Negatif')
            plt.plot(sentiment_percentages.index, sentiment_percentages[1], label='Sentimen Netral')
            plt.plot(sentiment_percentages.index, sentiment_percentages[2], label='Sentimen Positif')
            plt.title('Analisis 1 Minggu Terakhir')
            plt.xlabel('Tanggal')
            plt.ylabel('Persentase')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)


        else:
            st.write("Please go to the 'Preprocessing' tab and start the Preprocessing first.")
    else:
        st.write("Please upload a CSV file in the 'Upload Data' tab to get started.")
