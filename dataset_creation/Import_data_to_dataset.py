import json
import os
import sqlite3

def create_movies_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Movies (
            ImdbId TEXT PRIMARY KEY,
            Name TEXT,
            Genre TEXT,
            Age_Classification TEXT
        )
    ''')

def load_movies_from_directory(json_directory, db_path):
    # Conectar ao banco de dados SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Certificar-se de que a tabela existe com a coluna correta
    create_movies_table(cursor)

    inserted_count = 0
    skipped_count = 0

    # Percorrer todos os arquivos JSON no diretório
    for filename in os.listdir(json_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(json_directory, filename)

            # Abrir e carregar o conteúdo JSON do arquivo
            with open(file_path, 'r', encoding='utf-8') as file:
                movies_data = json.load(file)

                # Verificar se o JSON é uma lista de filmes
                if isinstance(movies_data, list):
                    # Itera sobre cada filme na lista
                    for movie_data in movies_data:
                        inserted, skipped = insert_movie(cursor, movie_data)
                        inserted_count += inserted
                        skipped_count += skipped
                else:
                    # Se não for uma lista, assume que é um único filme
                    inserted, skipped = insert_movie(cursor, movies_data)
                    inserted_count += inserted
                    skipped_count += skipped

    # Salvar as alterações e fechar a conexão
    print(f"Foram inseridos {inserted_count} filmes.")
    print(f"Nao foram inseridos {skipped_count} filmes (já existiam no banco de dados).")
    conn.commit()
    conn.close()

def insert_movie(cursor, movie_data):
    # Extrair os dados necessários
    imdb_id = movie_data.get("ImdbId")
    name = movie_data.get("name")
    genre = ", ".join(movie_data.get("genre", []))  # Converter lista de gêneros para string
    age_classification = movie_data.get("certificate")

    # Verificar se o filme já existe na tabela usando o ImdbId
    cursor.execute('SELECT 1 FROM Movies WHERE ImdbId = ?', (imdb_id,))
    exists = cursor.fetchone()

    # Inserir o filme apenas se ele não estiver no banco de dados
    if not exists:
        cursor.execute('''
            INSERT INTO Movies (ImdbId, Name, Genre, Age_Classification)
            VALUES (?, ?, ?, ?)
        ''', (imdb_id, name, genre, age_classification))
        return 1, 0  # Contabiliza como inserido
    else:
        return 0, 1  # Contabiliza como não inserido (duplicado)






if __name__ == "__main__":

    json_directory = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\Dataset\international-movies-json"  # Diretório com os arquivos JSON dos filmes
    db_path = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\Dataset\movies-and-reviews.sqlite3"  # Caminho para o banco de dados SQLite
    load_movies_from_directory(json_directory, db_path)
