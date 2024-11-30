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

def train_random_forest_binary():
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

    # Treinar modelo Random Forest
    print("Training Random Forest model for binary classification...")
    rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_model.fit(X_train_vec, y_train)

    # Fazer previsões
    y_pred = rf_model.predict(X_test_vec) 

    # Avaliar modelo
    print("Evaluating the model...")
    accuracy = accuracy_score(y_test, y_pred) 
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["not Adultos", "Adultos"]))

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Salvar o modelo e hiperparâmetros
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_directory = os.getcwd()
    model_folder = os.path.join(current_directory, f"ML_Program/saved_models/random_forest_{timestamp}")
    os.makedirs(model_folder, exist_ok=True)
    print(f"Saving model to folder: {model_folder}")

    # Define the save path
    model_path = os.path.join(model_folder, "random_forest_model.pkl")
    hyperparams_path = os.path.join(model_folder, "random_forest_hyperparameters.txt")

    # Save the model using joblib
    joblib.dump(rf_model, model_path)
    hyperparams = rf_model.get_params()  # Get model hyperparameters
    with open(hyperparams_path, "w") as f:
        f.write("Random Forest Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")

    print(f"Hyperparameters saved successfully at: {hyperparams_path}")
    print(f"Model saved successfully at: {model_path}")
