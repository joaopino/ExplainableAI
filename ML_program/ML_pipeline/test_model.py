from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from ML_pipeline.utilities import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing.database_setup.config import get_session
from data_preprocessing.database_setup.models import Comments, Films
from datetime import datetime
import time
import joblib
import os

def test_model(model_path):
    # Load data
    X_train_vec, X_test_vec, y_train_enc, y_test_enc, vectorizer, encoder, X_train, X_test = load_data()
    
    # Load model
    rf_model = joblib.load(model_path)
    
    # Vectorize data
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    
    # Make predictions
    y_pred = rf_model.predict(X_test_vec)
    
    # Convert one-hot encoded labels to binary format (if applicable)
    if y_test_enc.ndim > 1:  # If one-hot encoded
        y_test_enc = y_test_enc.argmax(axis=1)
    if y_pred.ndim > 1:  # If predictions are one-hot encoded
        y_pred = y_pred.argmax(axis=1)

    # Evaluate the model
    print("Evaluating the model...")
    accuracy = accuracy_score(y_test_enc, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

    # Generate and plot the confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

def test_binary_model(model_path):
    start_time = time.time()

    # Carregar dados da base de dados
    session = get_session()
    comments = session.query(Comments).all()
    session.close()

    # Preparar os dados
    X = [comment.preprocessed_comment for comment in comments if comment.preprocessed_comment]
    y = [session.query(Films).filter(Films.imdb_id == comment.imdb_id).one().age_rating for comment in comments if comment.preprocessed_comment]

    # Reclassify labels as binary: "Adultos" -> 1, all others -> 0
    print("Converting labels to binary: 'Adultos' vs. 'not Adultos'...")
    y_binary = [1 if label == "Adultos" else 0 for label in y]

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, stratify=y_binary, random_state=42)

    # Gerar embeddings com TF-IDF
    print("Generating embeddings using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)  # Limite de características para otimizar memória
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    
    
    
    
    
    # Load model
    rf_model = joblib.load(model_path)
    
    # Make predictions
    print("Making predictions...")
    y_pred = rf_model.predict(X_test_vec) 
    

    # Evaluate the model
    print("Evaluating the model...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["not Adultos", "Adultos"]))

    # Generate and plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["not Adultos", "Adultos"], yticklabels=["not Adultos", "Adultos"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.show()
