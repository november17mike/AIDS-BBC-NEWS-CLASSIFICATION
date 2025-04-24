import pandas as pd
import numpy as np
import nltk
import re
import tensorflow as tf
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('/content/news-article-categories.csv')

# Combine title and body
df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')

# Text preprocessing with lemmatization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2]
    return tokens

df['tokens'] = df['text'].apply(preprocess)

# Train Word2Vec model with more epochs and size
w2v_model = Word2Vec(
    sentences=df['tokens'],
    vector_size=200,
    window=5,
    min_count=2,
    workers=4,
    epochs=30
)

# Average embedding generator
def get_avg_embedding(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(200)

df['embedding'] = df['tokens'].apply(get_avg_embedding)
df = df[df['embedding'].apply(lambda x: np.any(x))]

# Prepare data
X = np.vstack(df['embedding'].values)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === IMPROVED CLASSIFIERS ===

# Random Forest with more trees
rf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Logistic Regression with regularization
lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# SVM with linear kernel
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# === IMPROVED NEURAL NETWORK ===
nn = Sequential([
    Dense(256, activation='relu', input_shape=(200,)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

nn.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
y_pred_nn = np.argmax(nn.predict(X_test), axis=1)

# === EVALUATION FUNCTION ===
def evaluate(name, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

evaluate("Random Forest", y_pred_rf)
evaluate("Logistic Regression", y_pred_lr)
evaluate("SVM", y_pred_svm)
evaluate("Neural Network", y_pred_nn)

# === PREDICTION FUNCTION FOR NEW ARTICLE ===
def predict_category(title, body, model_name='nn'):
    """
    Predict the category of a new news article using one of the trained models.

    Args:
        title (str): The news title
        body (str): The news body
        model_name (str): One of 'rf', 'lr', 'svm', 'nn'

    Returns:
        str: Predicted category name
    """
    # Combine and preprocess
    text = title + ' ' + body
    tokens = preprocess(text)

    if not tokens:
        return "Text too short or contains no valid tokens."

    # Get embedding
    embedding = get_avg_embedding(tokens).reshape(1, -1)

    # Predict with selected model
    if model_name == 'rf':
        pred = rf.predict(embedding)
    elif model_name == 'lr':
        pred = lr.predict(embedding)
    elif model_name == 'svm':
        pred = svm.predict(embedding)
    elif model_name == 'nn':
        pred = np.argmax(nn.predict(embedding), axis=1)
    else:
        return "Invalid model name. Use 'rf', 'lr', 'svm', or 'nn'."

    return label_encoder.inverse_transform(pred)[0]

# Example article
title = "Tech giants report record profits amid market rebound"
body = "Technology companies like Apple and Google reported significant profits this quarter, surprising analysts..."

# Predict using neural network
predicted_category = predict_category(title, body, model_name='nn')
print("Predicted Category:", predicted_category)
