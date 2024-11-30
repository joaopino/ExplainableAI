import os
import tensorflow as tf
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import joblib
from ML_pipeline.utilities import load_data
import numpy as np
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
from ML_pipeline.utilities import load_data
from data_preprocessing.database_setup.config import get_session
from data_preprocessing.database_setup.models import Comments, Films
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def explainable_nn(model_path):
    # Load the trained Keras model
    nn_model = load_model(model_path)
    print("Neural Network model loaded successfully.")

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

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, stratify=y_binary, random_state=42)

    # Generate embeddings using TF-IDF
    print("Generating embeddings using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=768)  # Ensure this matches the model's input shape
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()


    # Initialize LIME Tabular Explainer
    print("Initializing LIME explainer...")
    explainer = LimeTabularExplainer(
        X_train_vec,
        training_labels=y_train,
        feature_names=vectorizer.get_feature_names_out(),
        class_names=["not Adultos", "Adultos"],
        mode="classification"
    )

    # Select a sample for explanation
    instance_index = 0  # Change this index to analyze different test samples
    instance = X_test_vec[instance_index].reshape(1, -1)
    print(f"Explaining prediction for instance {instance_index}...")

    # Define a prediction function compatible with LIME
    def predict_proba(input_data):
        input_data = np.array(input_data).astype(np.float32)  # Ensure correct data type
        return nn_model.predict(input_data)

    # Generate explanation for the selected instance
    explanation = explainer.explain_instance(
        instance[0], 
        predict_proba,  # Neural network prediction function
        num_features=10
    )

    # Extract top features
    feature_importance = explanation.as_list()
    print("Top features and contributions:", feature_importance)

    # Generate highlighted text
    original_text = X_test[instance_index]  # Get the original text
    highlighted_text = original_text
    for word, weight in feature_importance:
        color = "red" if weight > 0 else "blue"
        highlighted_text = highlighted_text.replace(word, f"\\textcolor{{{color}}}{{{word}}}")

    # Save the explanation as a visualization
    explanation.save_to_file("lime_explanation_nn.html")
    print("Explanation saved as lime_explanation_nn.html")

    # Plot the highlighted text
    plt.figure(figsize=(10, 7))
    plt.text(0.5, 0.5, highlighted_text, fontsize=12, wrap=True, horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    plt.title("Highlighted Text with Important Features")
    plt.savefig("highlighted_text_nn.png")
    print("Highlighted text saved as highlighted_text_nn.png")
