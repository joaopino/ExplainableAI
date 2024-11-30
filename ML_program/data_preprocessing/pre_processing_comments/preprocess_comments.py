import re
import spacy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from ML_program.data_preprocessing.database_setup.config import engine
from ML_program.data_preprocessing.database_setup.models import Comments

# Inicializar spaCy
nlp = spacy.load("en_core_web_sm")

# Configuração da sessão SQLAlchemy
Session = sessionmaker(bind=engine)
session = Session()


# Funções de Pré-processamento

def clean_text(text):
    """
    Cleans text by removing punctuation and special characters, and converting to lowercase.

    Args:
        text (str): The original text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()


def tokenize_and_filter(text):
    """
    Tokenizes the cleaned text and filters out stopwords and non-alphabetic tokens.

    Args:
        text (str): The cleaned text to be tokenized.

    Returns:
        list: A list of lemmatized tokens without stopwords.
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]


def preprocess_comment(comment):
    """
    Conducts the entire preprocessing pipeline: cleaning, tokenizing, and lemmatizing.

    Args:
        comment (str): The original comment text.

    Returns:
        str: The fully preprocessed text.
    """
    cleaned_text = clean_text(comment)
    tokens = tokenize_and_filter(cleaned_text)
    return ' '.join(tokens)


# Funções para Modificação da Base de Dados:

def save_preprocessed_comment(comment_id, preprocessed_text):
    """
    Saves the preprocessed comment text into the `preprocessed_comment` column for a specific comment.

    Args:
        comment_id (int): The ID of the comment to be updated.
        preprocessed_text (str): The preprocessed comment text.
    """
    comment = session.query(Comments).filter_by(comment_id=comment_id).one_or_none()
    if comment:
        comment.preprocessed_comment = preprocessed_text
        try:
            session.commit()
            print(f"Updated comment {comment_id} with preprocessed text.")
        except IntegrityError as e:
            print(f"Error updating comment {comment_id}: {e}")
            session.rollback()


# Função para Processar e Atualizar Todos os Comentários

def preprocess_and_save_all_comments():
    """
    Fetches all comments from the database, preprocesses each, and updates the respective record
    in the `preprocessed_comment` column.
    """
    comments = session.query(Comments).all()

    for comment in comments:
        preprocessed_comment = preprocess_comment(comment.comment)
        save_preprocessed_comment(comment.comment_id , preprocessed_comment)


# Função Principal

def main():
    """
    Executes the full preprocessing and database update pipeline:
    1. Ensures the database has a column for preprocessed comments.
    2. Fetches, preprocesses, and updates each comment in the database.
    """
    preprocess_and_save_all_comments()


if __name__ == "__main__":
    main()
