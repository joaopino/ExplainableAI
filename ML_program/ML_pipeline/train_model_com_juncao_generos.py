import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer, TFBertModel
from ML_program.data_preprocessing.database_setup.models import Films, Comments
from ML_program.data_preprocessing.database_setup.config import get_session
import tensorflow as tf
import logging

# Configurar logs para evitar mensagens excessivas
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# Função para obter embeddings dos comentários usando BERT
def get_comment_embeddings(comments):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    # Preparar os dados para o modelo BERT
    inputs = tokenizer(comments, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = model(inputs)

    # Média dos embeddings (última camada oculta)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings.numpy()


def train_model():
    # Conectar à base de dados
    session = get_session()

    # Obter dados de filmes e comentários
    films = session.query(Films).all()
    comments = session.query(Comments).all()

    session.close()

    # Criar dicionário de comentários agrupados por imdb_id
    comments_by_imdb = {}
    for comment in comments:
        if comment.imdb_id in comments_by_imdb:
            comments_by_imdb[comment.imdb_id].append(comment.preprocessed_comment)
        else:
            comments_by_imdb[comment.imdb_id] = [comment.preprocessed_comment]

    # Preparar os gêneros
    all_genres = set(
        genre.strip()
        for film in films
        for genre in (film.genre or "").split(",")
        if genre.strip()  # Ignorar gêneros vazios
    )
    mlb = MultiLabelBinarizer(classes=sorted(all_genres))
    genres_matrix = mlb.fit_transform([
        set(genre.strip() for genre in (film.genre or "").split(",") if genre.strip())
        for film in films
    ])

    # Preparar as classes (age_rating)
    age_ratings = [film.age_rating for film in films]
    encoder = LabelEncoder()
    age_ratings_enc = to_categorical(encoder.fit_transform(age_ratings))

    # Gerar embeddings dos comentários usando BERT
    comments_embeddings = []
    for film in films:
        if film.imdb_id in comments_by_imdb:
            film_comments = comments_by_imdb[film.imdb_id]
            embeddings = get_comment_embeddings(film_comments)
            comments_embeddings.append(np.mean(embeddings, axis=0))  # Média dos embeddings
        else:
            comments_embeddings.append(np.zeros(768))  # Embedding de zeros se não houver comentários

    comments_embeddings = np.array(comments_embeddings)

    # Combinar gêneros e embeddings de comentários
    X = np.hstack([comments_embeddings, genres_matrix])
    y = age_ratings_enc

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Criar o modelo
    input_comments = Input(shape=(768,))
    input_genres = Input(shape=(X.shape[1] - 768,))

    x = Dense(256, activation="relu")(input_comments)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Concatenar com os gêneros
    combined = Concatenate()([x, input_genres])
    output = Dense(y.shape[1], activation="softmax")(combined)

    model = Model(inputs=[input_comments, input_genres], outputs=output)

    # Compilar o modelo
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Treinar o modelo
    history = model.fit([X_train[:, :768], X_train[:, 768:]], y_train, epochs=30, batch_size=32, validation_split=0.2)

    # Avaliar o modelo
    test_loss, test_accuracy = model.evaluate([X_test[:, :768], X_test[:, 768:]], y_test)
    print(f"Loss no teste: {test_loss:.4f}, Acurácia no teste: {test_accuracy:.4f}")

    return history, test_loss, test_accuracy


if __name__ == "__main__":
    train_model()
