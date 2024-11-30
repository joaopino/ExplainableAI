from data_preprocessing.database_setup.config import get_session
from data_preprocessing.database_setup.models import Comments, Films
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_data():
    print("Loading data from the database...")
    session = get_session()
    comments = session.query(Comments).all()
    session.close()

    # Prepare the data
    X = [comment.preprocessed_comment for comment in comments if comment.preprocessed_comment]
    y = [
        session.query(Films)
        .filter(Films.imdb_id == comment.imdb_id)
        .one()
        .age_rating
        for comment in comments
        if comment.preprocessed_comment
    ]

    # Split into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    # Generate embeddings with TF-IDF
    print("Generating embeddings using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to optimize memory
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    # Encode labels
    print("Encoding labels...")
    encoder = LabelEncoder()
    y_train_enc = to_categorical(encoder.fit_transform(y_train))
    y_test_enc = to_categorical(encoder.transform(y_test))
    
    return X_train_vec, X_test_vec, y_train_enc, y_test_enc, vectorizer, encoder, X_train, X_test