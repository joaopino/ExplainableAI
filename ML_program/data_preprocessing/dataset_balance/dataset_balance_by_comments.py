import sqlite3
import random
import os

# Caminho para os bancos de dados
input_db_path = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\ML_program\all_films.db"
output_db_path = "balanced_films_by_commentsNumber.db"

# Remover o arquivo de saída, caso já exista
if os.path.exists(output_db_path):
    os.remove(output_db_path)

# Conectar à base de dados de entrada
input_conn = sqlite3.connect(input_db_path)
input_cursor = input_conn.cursor()

# Criar a base de dados balanceada
output_conn = sqlite3.connect(output_db_path)
output_cursor = output_conn.cursor()

# Criar tabelas no banco de dados balanceado
input_cursor.execute("PRAGMA table_info(comments)")
comments_columns = input_cursor.fetchall()
comments_schema = ", ".join([f"{col[1]} {col[2]}" for col in comments_columns])
output_cursor.execute(f"CREATE TABLE comments ({comments_schema})")

input_cursor.execute("PRAGMA table_info(films)")
films_columns = input_cursor.fetchall()
films_schema = ", ".join([f"{col[1]} {col[2]}" for col in films_columns])
output_cursor.execute(f"CREATE TABLE films ({films_schema})")

# Obter as categorias de age_rating
input_cursor.execute("SELECT DISTINCT age_rating FROM films")
categories = [row[0] for row in input_cursor.fetchall()]

# Encontrar o menor número de comentários por categoria
comment_counts = {}
for category in categories:
    query = """
        SELECT COUNT(*)
        FROM comments
        WHERE imdb_id IN (SELECT imdb_id FROM films WHERE age_rating = ?)
    """
    input_cursor.execute(query, (category,))
    comment_counts[category] = input_cursor.fetchone()[0]

min_comments = min(comment_counts.values())

# Selecionar e copiar dados balanceados para o novo banco de dados
for category in categories:
    # Obter IDs de filmes por categoria
    input_cursor.execute("SELECT imdb_id FROM films WHERE age_rating = ?", (category,))
    imdb_ids = [row[0] for row in input_cursor.fetchall()]

    # Obter comentários balanceados
    query = "SELECT * FROM comments WHERE imdb_id = ?"
    selected_comments = []
    for imdb_id in imdb_ids:
        input_cursor.execute(query, (imdb_id,))
        comments = input_cursor.fetchall()
        if comments:
            selected_comments.extend(comments)

    # Balancear comentários
    balanced_comments = random.sample(selected_comments, min_comments)
    balanced_imdb_ids = set([comment[1] for comment in balanced_comments])  # Supondo que imdb_id é a segunda coluna

    # Copiar filmes correspondentes
    films_query = f"SELECT * FROM films WHERE imdb_id IN ({', '.join(['?'] * len(balanced_imdb_ids))})"
    input_cursor.execute(films_query, list(balanced_imdb_ids))
    films = input_cursor.fetchall()
    output_cursor.executemany(f"INSERT INTO films VALUES ({', '.join(['?'] * len(films_columns))})", films)

    # Copiar comentários balanceados
    output_cursor.executemany(f"INSERT INTO comments VALUES ({', '.join(['?'] * len(comments_columns))})", balanced_comments)

# Salvar e fechar as conexões
output_conn.commit()
input_conn.close()
output_conn.close()

print(f"Dados balanceados foram salvos em {output_db_path}.")
