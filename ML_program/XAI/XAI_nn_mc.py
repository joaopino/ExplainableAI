import os
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
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

import os
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def explainable_nn_multiclass(model_path):
    # Load the trained Keras model
    nn_model = load_model(model_path)
    print("Neural Network model loaded successfully.")

    # Prepare data (simulate loading preprocessed data)
    session = get_session()
    comments = session.query(Comments).all()
    session.close()

    # Extract features and labels
    X = [comment.preprocessed_comment for comment in comments if comment.preprocessed_comment]
    y = [session.query(Films).filter(Films.imdb_id == comment.imdb_id).one().age_rating for comment in comments if comment.preprocessed_comment]

    # Encode target labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)  # One-hot encoding for multiclass classification

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, stratify=y_encoded, random_state=42)

    # Transform text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=768)  # Ensure this matches the model's input shape
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    # Initialize LIME Tabular Explainer
    print("Initializing LIME explainer...")
    explainer = LimeTabularExplainer(
        X_train_vec,
        training_labels=y_encoded,
        feature_names=vectorizer.get_feature_names_out(),
        class_names=label_encoder.classes_,
        mode="classification"
    )

    # Select a sample for explanation
    instance_index = 10  # Change this index to analyze different samples
    instance = X_test_vec[instance_index].reshape(1, -1)
    print(f"Explaining prediction for instance {instance_index}...")

    # Define a prediction function for the neural network
    def predict_proba(input_data):
        return nn_model.predict(input_data)

    # Generate explanation for the selected instance
    explanation = explainer.explain_instance(
        instance[0],  # Input sample
        predict_proba,  # Prediction function
        num_features=10  # Number of features to include in the explanation
    )

    # Save and visualize explanation
    explanation.save_to_file("lime_explanation_multiclass.html")
    print("Explanation saved as lime_explanation_multiclass.html")

    # Highlight important features for the input text
    feature_importance = explanation.as_list()
    original_text = X_test[instance_index]
    highlighted_text = original_text
    for word, weight in feature_importance:
        color = "red" if weight > 0 else "blue"
        highlighted_text = highlighted_text.replace(word, f"\\textcolor{{{color}}}{{{word}}}")

    # Plot highlighted text
    plt.figure(figsize=(10, 7))
    plt.text(0.5, 0.5, highlighted_text, fontsize=12, wrap=True, horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    plt.title("Highlighted Text with Important Features")
    plt.savefig("highlighted_text_multiclass.png")
    print("Highlighted text saved as highlighted_text_multiclass.png")
