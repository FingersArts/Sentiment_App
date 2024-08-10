import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from preprocessing import preprocess_and_calculate_tfidf, LogisticRegression, preprocess_input
from preprocessing import plot_confusion_matrix, k_fold_cross_validation
import matplotlib.pyplot as plt

# Title of the application
st.title('Analisis Sentimen dengan Multinomial Logistic Regression')

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload Data", "Preprocessing", "TF-IDF", "Multinomial Logistic Regression", "Prediksi Sentimen"])

# Tab for uploading data
with tab1:
    # Upload CSV file
    uploaded_file = st.file_uploader("Pilih File CSV", type=["csv"])

    # If a file is uploaded
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Store the original dataframe in session state
        st.session_state['df'] = df

        # Display the original DataFrame
        st.subheader('Data Asli')
        st.write(df)

# Tab for pre-processing
with tab2:
    if uploaded_file is not None:
        # Button to start preprocessing
        if st.button('Mulai Preprocessing'):
            # Process the dataframe
            preprocessed_df, ranking = preprocess_and_calculate_tfidf(st.session_state['df'])
            
            # Store preprocessed data and TF-IDF ranking in session state
            st.session_state['preprocessed_df'] = preprocessed_df
            st.session_state['ranking'] = ranking
            
            st.subheader('Data Setelah Preprocessing')
            st.write(preprocessed_df[['text', 'casefolding', 'cleanedtext', 'slangremoved', 'stopwordremoved', 'stemmedtext', 'tokenize']])

            # Count word frequencies in the 'stemmedtext' column
            word_counts = preprocessed_df['stemmedtext'].str.split(expand=True).stack().value_counts()
            
            # Visualize the top 10 most frequent words
            st.subheader('Visualisasi 10 Kata Teratas')
            st.bar_chart(word_counts.head(10))

            # Display the top 10 most frequent words
            st.subheader('10 Kata Teratas')
            st.write(word_counts.head(10))
    else:
        st.write("Pilih File CSV di tab 'Upload Data' untuk Memulai.")

# Tab for TF-IDF results
with tab3:
    if uploaded_file is not None:
        if 'preprocessed_df' in st.session_state:
            # Display all TF-IDF ranked terms
            st.subheader('TF-IDF')
            st.dataframe(st.session_state['ranking'])
        else:
            st.write("Mulai tahap Preprocessing terlebih dahulu")
    else:
        st.write("Pilih File CSV di tab 'Upload Data' untuk Memulai.")

# Tab for Logistic Regression
with tab4:
    if uploaded_file is not None:
        if 'preprocessed_df' in st.session_state:
            # Map sentiment_val to label
            sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
            st.session_state['df']['label'] = st.session_state['df']['sentiment_val'].map(sentiment_mapping)

            # Prepare data for Logistic Regression
            words = ' '.join(st.session_state['preprocessed_df']['stemmedtext']).split()
            word_counts = Counter(words)
            features = [word for word, count in word_counts.most_common(1000)]

            def transform_to_features(texts, features):
                def count_features(words):
                    return [words.count(feature) for feature in features]
                return np.array([count_features(text.split()) for text in texts])

            X = transform_to_features(st.session_state['preprocessed_df']['stemmedtext'], features)
            y = st.session_state['df']['label'].values.astype(int)

            # Store features and labels in session state
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['features'] = features

            # Initialize and train the model
            model = LogisticRegression(num_iter=200, learning_rate=0.44)
            (mean_accuracy, mean_precision, mean_recall, average_confusion, avg_class_accuracies, avg_precision_per_class, avg_recall_per_class) = k_fold_cross_validation(X, y, model, k=10)
            
            # Store the trained model in session state
            st.session_state['model'] = model

            # Display model performance and other details...

            # Display model performance
            st.subheader('Performa Model')

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Average Accuracies per Class:")
                st.markdown(f"**Sentiment negative**: {avg_class_accuracies[0]:.6f}")
                st.markdown(f"**Sentiment neutral**: {avg_class_accuracies[1]:.6f}")
                st.markdown(f"**Sentiment positive**: {avg_class_accuracies[2]:.6f}")

            with col2:
                st.markdown("### Average Precision per Class:")
                st.markdown(f"**Sentiment negative**: {avg_precision_per_class[0]:.6f}")
                st.markdown(f"**Sentiment neutral**: {avg_precision_per_class[1]:.6f}")
                st.markdown(f"**Sentiment positive**: {avg_precision_per_class[2]:.6f}")

            with col3:
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
            st.session_state['df']['date'] = pd.to_datetime(st.session_state['df']['date'])
            st.session_state['df']['tanggal'] = st.session_state['df']['date'].dt.strftime('%Y-%m-%d')

            # Filter data untuk 1 minggu terakhir
            df_last_week = st.session_state['df'][st.session_state['df']['date'] >= st.session_state['df']['date'].max() - pd.to_timedelta(7, unit='D')]

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
            st.write("Mulai tahap Preprocessing terlebih dahulu")
    else:
        st.write("Pilih File CSV di tab 'Upload Data' untuk Memulai.")

with tab5:
    if uploaded_file is not None:
        if 'preprocessed_df' in st.session_state:
            st.subheader('Prediksi Sentimen')
            # Text input from the user
            user_input = st.text_area("Masukan Kalimat")

            if st.button('Prediksi Sentimen'):
                if 'model' in st.session_state:
                    model = st.session_state['model']

                    # Predict sentiment for user input
                    def predict_sentiment(text, model, features):
                        preprocessed_text = preprocess_input(text)
                        feature_vector = transform_to_features([preprocessed_text], features)
                        prediction = model.predict(feature_vector)
                        return prediction[0]
                    
                    # Preprocess the input and predict sentiment
                    predicted_sentiment = predict_sentiment(user_input, model, st.session_state['features'])
                    
                    # Map predicted sentiment index to sentiment label
                    sentiment_labels = ["negative", "neutral", "positive"]
                    st.write(f"Prediksi Sentimen: **{sentiment_labels[predicted_sentiment]}**")
                else:
                    st.write("Model belum siap, lakukan proses Logistic Regression terlebih dahulu.")
            else:
                st.write("Masukan Kalimat untuk memulai Prediksi Sentimen.")
        else:
            st.write("Mulai tahap Preprocessing terlebih dahulu")
    else:
        st.write("Pilih File CSV di tab 'Upload Data' untuk Memulai.")
