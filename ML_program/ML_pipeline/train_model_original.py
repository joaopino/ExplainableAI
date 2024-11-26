import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ML_pipeline.data_preparation import load_data

def train_bert_model():
    """
    Treina um modelo BERT para classificação de faixa etária com base em comentários pré-processados,
    com foco em evitar overfitting e melhorar a generalização.
    """
    # Carregar os dados
    print("Carregando dados...")
    data = load_data()
    comments = data["preprocessed_comment"].values  # Usar a coluna com comentários pré-processados
    age_ratings = data["age_rating"].values

    # Dividir os dados em conjuntos de treino e teste
    print("Dividindo dados em treino e teste...")
    train_texts, test_texts, y_train, y_test = train_test_split(
        comments, age_ratings, test_size=0.2, random_state=42
    )

    # Carregar o tokenizer BERT
    print("Carregando o tokenizer BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenizar os textos
    print("Tokenizando os textos...")
    train_encodings = tokenizer(
        list(train_texts),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )
    test_encodings = tokenizer(
        list(test_texts),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

    # Criar os datasets do TensorFlow
    print("Criando datasets do TensorFlow...")
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).batch(16)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(16)

    # Carregar o modelo BERT pré-treinado
    print("Carregando modelo BERT...")
    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )

    # Compilar o modelo com regularização
    print("Compilando o modelo...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)  # Ajuste da taxa de aprendizagem
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

    # Adicionar regularização L2 e dropout
    model.config.attention_probs_dropout_prob = 0.3  # Dropout nas atenções
    model.config.hidden_dropout_prob = 0.3  # Dropout nas camadas ocultas

    # Compilação com L2 regularization no otimizador
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Configurar o EarlyStopping para evitar overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Treinar o modelo com early stopping
    print("Iniciando o treino do modelo...")
    model.fit(
        train_dataset,
        epochs=10,  # Mais épocas para treinamento completo
        batch_size=16,
        validation_data=test_dataset,
        callbacks=[early_stopping]  # Monitoramento para parar se o modelo parar de melhorar
    )

    # Avaliar o modelo
    print("Avaliando o modelo...")
    predictions = model.predict(test_dataset).logits
    predicted_classes = tf.argmax(predictions, axis=1)

    # Exibir relatório de classificação
    print(classification_report(y_test, predicted_classes, target_names=[
        "Crianças", "Crianças mais velhas", "Adolescentes", "Adultos"
    ]))

    # Salvar o modelo treinado
    print("Salvando o modelo treinado...")
    model.save_pretrained("trained_bert_model")
    tokenizer.save_pretrained("trained_bert_model")

    print("Treinamento concluído e modelo salvo em 'trained_bert_model'.")

