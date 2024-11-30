import os
import sqlite3
import re


def inserir_reviews_no_sqlite(db_path='reviews.db'):
    # Criar a ligação à base de dados
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Criar a tabela se ainda não existir
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Reviews (
        ReviewID INTEGER PRIMARY KEY,
        ImdbId TEXT,
        ReviewText TEXT,
        Sentiment TEXT
    );
    ''')

    # Função para extrair o ImdbId do URL
    def extract_imdb_id(url):
        match = re.search(r'tt\d+', url)
        return match.group(0) if match else None

    # Diretórios e respetivos ficheiros de URLs
    data_structure = {
        'pos': {'sentiment': 'positive', 'urls_file': 'urls_pos.txt'},
        'neg': {'sentiment': 'negative', 'urls_file': 'urls_neg.txt'},
        'unsup': {'sentiment': 'unsupervised', 'urls_file': 'urls_unsup.txt'}
    }

    # Iterar pelos diretórios e ficheiros de URLs
    for folder, info in data_structure.items():
        sentiment_label = info['sentiment']
        urls_file = info['urls_file']

        # Caminho para o ficheiro de URLs
        urls_path = os.path.join(urls_file)

        # Ler os URLs e extrair os ImdbIds
        with open(urls_path, 'r') as f:
            imdb_ids = [extract_imdb_id(line.strip()) for line in f.readlines()]

        # Caminho para a pasta de reviews
        reviews_dir = os.path.join(folder)

        # Ler cada ficheiro de review e inserir na tabela
        for i, review_file in enumerate(sorted(os.listdir(reviews_dir))):
            if review_file.endswith('.txt'):
                with open(os.path.join(reviews_dir, review_file), 'r', encoding='utf-8') as rf:
                    review_text = rf.read().strip()

                    # Inserir os dados na tabela
                    cursor.execute('''
                        INSERT INTO Reviews (ImdbId, ReviewText, Sentiment)
                        VALUES (?, ?, ?)
                    ''', (imdb_ids[i], review_text, sentiment_label))

    # Confirmar e fechar a ligação
    conn.commit()
    conn.close()

    print("Reviews e IDs inseridos na tabela com sucesso.")


# Chamar a função
inserir_reviews_no_sqlite()
