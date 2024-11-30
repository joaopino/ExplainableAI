import sqlite3
import random
import os

# Caminho para os bancos de dados
input_db_path = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\ML_program\all_films.db"
output_db_path = "balanced_films_by_filmsNumber.db"

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

# Encontrar o menor número de filmes por categoria
film_counts = {}
for category in categories:
    input_cursor.execute("SELECT COUNT(*) FROM films WHERE age_rating = ?", (category,))
    film_counts[category] = input_cursor.fetchone()[0]

min_films = min(film_counts.values())

# Selecionar e copiar dados balanceados para o novo banco de dados
for category in categories:
    # Balancear filmes
    input_cursor.execute("SELECT * FROM films WHERE age_rating = ?", (category,))
    films = input_cursor.fetchall()
    balanced_films = random.sample(films, min_films)
    output_cursor.executemany(f"INSERT INTO films VALUES ({', '.join(['?'] * len(films_columns))})", balanced_films)

    # Balancear comentários associados aos filmes selecionados
    balanced_film_ids = [film[0] for film in balanced_films]  # Supondo que o ID do filme é a primeira coluna
    comments_query = f"SELECT * FROM comments WHERE imdb_id IN ({', '.join(['?'] * len(balanced_film_ids))})"
    input_cursor.execute(comments_query, balanced_film_ids)
    comments = input_cursor.fetchall()
    output_cursor.executemany(f"INSERT INTO comments VALUES ({', '.join(['?'] * len(comments_columns))})", comments)

# Salvar e fechar as conexões
output_conn.commit()
input_conn.close()
output_conn.close()

print(f"Dados balanceados foram salvos em {output_db_path}.")
