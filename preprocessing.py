import pandas as pd
import re
import nltk
import requests
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import ast

# Download required NLTK resources
nltk.download(['stopwords', 'punkt', 'wordnet'])
nltk.download('punkt_tab')

# Initialize the stemmer outside the stemming function
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Define text cleaning function
def cleaning_text(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    text = re.sub(r'@[\w]*', ' ', text)
    text = re.sub(r'\brt\b', ' ', text, flags=re.IGNORECASE)
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, " ")
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Muat kamus slangword dari file CSV
kamus_slang = pd.read_csv("kamus-slang.csv", header=None, names=['slang', 'formal'])
lookup_dict = dict(zip(kamus_slang['slang'], kamus_slang['formal']))
# Mengganti kata slang dengan kata asli
def slangremove(text, lookup_dict=lookup_dict):
    words = text.split()
    new_words = [lookup_dict.get(word, word) for word in words]
    return ' '.join(new_words)

# Define stopwords
sastrawi_stopword = "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/sastrawi-stopwords.txt"
stopwords_l = stopwords.words('indonesian')
response = requests.get(sastrawi_stopword)
stopwords_l += response.text.split('\n')

custom_st = '''
yg yang dgn ane smpai bgt gua gwa si tu ama utk udh btw
ntar lol ttg emg aj aja tll sy sih kalo nya trsa mnrt nih
'''
st_words = set(stopwords_l)
custom_stopword = set(custom_st.split())
stop_words = st_words | custom_stopword

# Define function to remove stopwords
def remove_stopword(text, stop_words=stop_words):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

# Define stemming function
def stemming(text):
    return stemmer.stem(text)

# Define tokenization function
def tokenize(text):
    return word_tokenize(text)

# Define functions for TF-IDF calculation
def convert_text_list(texts):
    if isinstance(texts, list):
        return texts
    return ast.literal_eval(texts)

def calc_TF(document):
    TF_dict = {}
    for term in document:
        TF_dict[term] = TF_dict.get(term, 0) + 1
    for term in TF_dict:
        TF_dict[term] /= len(document)
    return TF_dict

def calc_DF(tfDict):
    count_DF = {}
    for document in tfDict:
        for term in document:
            count_DF[term] = count_DF.get(term, 0) + 1
    return count_DF

def calc_IDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
    return IDF_Dict

def calc_TF_IDF(TF, IDF):
    TF_IDF_Dict = {}
    for key in TF:
        TF_IDF_Dict[key] = TF[key] * IDF[key]
    return TF_IDF_Dict

def calc_TF_IDF_Vec(__TF_IDF_Dict, unique_term):
    TF_IDF_vector = [0.0] * len(unique_term)
    for i, term in enumerate(unique_term):
        if term in __TF_IDF_Dict:
            TF_IDF_vector[i] = __TF_IDF_Dict[term]
    return TF_IDF_vector

# Main function to process the dataframe
def preprocess_and_calculate_tfidf(df):
    df2 = df.drop_duplicates(subset=["text"])
    df3 = df2[df2['text'].str.len() >= 4]
    df3['casefolding'] = df3['text'].str.lower()
    df3['cleanedtext'] = df3['casefolding'].apply(cleaning_text)
    df3['slangremoved'] = df3['cleanedtext'].apply(lambda x: slangremove(x, lookup_dict))
    df3['stopwordremoved'] = df3['slangremoved'].apply(remove_stopword)
    df3['stemmedtext'] = df3['stopwordremoved'].apply(stemming)
    df3['tokenize'] = df3['stemmedtext'].apply(tokenize)
    
    df3["text_list"] = df3["tokenize"].apply(convert_text_list)
    df3["TF_dict"] = df3["text_list"].apply(calc_TF)
    DF = calc_DF(df3["TF_dict"])
    n_document = len(df3)
    IDF = calc_IDF(n_document, DF)
    df3["TF-IDF_dict"] = df3["TF_dict"].apply(lambda tf: calc_TF_IDF(tf, IDF))
    
    sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:50]
    unique_term = [item[0] for item in sorted_DF]
    
    df3["TF_IDF_Vec"] = df3["TF-IDF_dict"].apply(lambda tfidf: calc_TF_IDF_Vec(tfidf, unique_term))
    
    TF_IDF_Vec_List = np.array(df3["TF_IDF_Vec"].to_list())
    sums = TF_IDF_Vec_List.sum(axis=0)
    
    data = []
    for col, term in enumerate(unique_term):
        data.append((term, sums[col]))
    
    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    ranking = ranking.sort_values('rank', ascending=False)
    
    return df3, ranking

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Logistic Regression Model
class LogisticRegression:
    def __init__(self, num_iter, learning_rate):
        self.num_iter = num_iter
        self.learning_rate = learning_rate

    def compute_cost(self, X, y):
        num_samples = X.shape[0]
        scores = np.dot(X, self.weights) + self.bias
        probs = softmax(scores)
        correct_logprobs = -np.log(probs[range(num_samples), y])
        cost = np.sum(correct_logprobs) / num_samples
        return cost, probs

    def compute_gradients(self, X, y, probs):
        num_samples = X.shape[0]
        dscores = probs
        dscores[range(num_samples), y] -= 1
        dscores /= num_samples
        dweights = np.dot(X.T, dscores)
        dbias = np.sum(dscores, axis=0)
        return dweights, dbias

    def train(self, X, y):
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros(num_classes)

        for i in range(self.num_iter):
            cost, probs = self.compute_cost(X, y)
            if i % 199 == 0:
                print('Iteration: %d, Cost: %f' % (i, cost))
            dweights, dbias = self.compute_gradients(X, y, probs)
            self.weights -= self.learning_rate * dweights
            self.bias -= self.learning_rate * dbias

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(scores, axis=1)

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Function to calculate precision
def calculate_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    precision = np.zeros(num_classes)
    for class_ in range(num_classes):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        false_positives = np.sum((y_true != class_) & (y_pred == class_))
        precision[class_] = true_positives / (true_positives + false_positives + 1e-10)
    return precision

# Function to calculate recall
def calculate_recall(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    recall = np.zeros(num_classes)
    for class_ in range(num_classes):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        false_negatives = np.sum((y_true == class_) & (y_pred != class_))
        recall[class_] = true_positives / (true_positives + false_negatives + 1e-10)
    return recall

# Function to calculate class-wise accuracy
def calculate_class_accuracy(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    class_accuracies = {}
    for class_ in range(num_classes):
        true_class = y_true == class_
        pred_class = y_pred == class_
        if np.sum(true_class) == 0:  # Avoid division by zero
            accuracy = np.nan
        else:
            accuracy = np.sum(true_class & pred_class) / np.sum(true_class)
        class_accuracies[class_] = accuracy
    return class_accuracies

# Function to calculate the confusion matrix
def calculate_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        true_index = y_true[i]
        pred_index = y_pred[i]
        confusion_matrix[true_index, pred_index] += 1
    return confusion_matrix

# Function to plot the confusion matrix
def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Average Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def k_fold_cross_validation(X, y, model, k=10):
    num_samples = X.shape[0]
    fold_size = num_samples // k
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    all_class_accuracies = []
    all_precision = []
    all_recall = []

    # Get the number of classes from the entire dataset
    num_classes = len(np.unique(y))
    total_cm = np.zeros((num_classes, num_classes))

    for i in range(k):
        test_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = np.concatenate((indices[:i*fold_size], indices[(i+1)*fold_size:]))
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = calculate_accuracy(y_test, y_pred)
        precision = calculate_precision(y_test, y_pred)
        recall = calculate_recall(y_test, y_pred)
        class_accuracies = calculate_class_accuracy(y_test, y_pred)
        confusion = calculate_confusion_matrix(y_test, y_pred, num_classes)

        total_cm += confusion
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        all_class_accuracies.append(class_accuracies)
        all_precision.append(precision)
        all_recall.append(recall)

    # Average accuracy per class
    avg_class_accuracies = {key: np.nanmean([acc.get(key, np.nan) for acc in all_class_accuracies]) for key in all_class_accuracies[0]}

    # Make sure all_precision and all_recall are arrays of the same length
    all_precision = np.array([np.pad(p, (0, num_classes - len(p)), 'constant', constant_values=np.nan) for p in all_precision])
    all_recall = np.array([np.pad(r, (0, num_classes - len(r)), 'constant', constant_values=np.nan) for r in all_recall])

    avg_precision_per_class = np.nanmean(all_precision, axis=0)
    avg_recall_per_class = np.nanmean(all_recall, axis=0)

    return (np.mean(accuracy_scores), avg_precision_per_class, avg_recall_per_class,
            total_cm / k, avg_class_accuracies, avg_precision_per_class, avg_recall_per_class)


# Function to preprocess user input
def preprocess_input(text):
    text = cleaning_text(text)
    text = slangremove(text)
    text = remove_stopword(text)
    text = stemming(text)
    return text
