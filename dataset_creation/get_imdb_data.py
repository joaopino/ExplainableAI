import os

from imdb import IMDb
from read_and_filter_IMDBID import run
import sqlite3

# Passo 1: Configurar a base de dados
def setup_database():
    conn = sqlite3.connect("imdb_filmes.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS filmes (
            imdb_id TEXT PRIMARY KEY,
            nome TEXT,
            genero TEXT,
            faixa_etaria TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

# Passo 2: Função para obter dados do IMDb com IMDbPY
def get_movie_info(imdb_id):
    ia = IMDb()
    movie = ia.get_movie(imdb_id[2:])  # Remover o prefixo 'tt' do ID

    nome = movie.get('title', 'N/A')
    generos = movie.get('genres', [])
    genero = ", ".join(generos) if generos else 'N/A'
    certificados = movie.get('certificates', [])

    # Filtrar a faixa etária americana
    faixa_etaria_americana = 'N/A'
    for cert in certificados:
        if "United States" in cert:
            faixa_etaria_americana = cert
            break

    return imdb_id, nome, genero, faixa_etaria_americana

# Passo 3: Inserir dados na base de dados
def insert_movie_data(cursor, movie_data):
    cursor.execute('''
        INSERT OR REPLACE INTO filmes (imdb_id, nome, genero, faixa_etaria) 
        VALUES (?, ?, ?, ?)
    ''', movie_data)

# Função para ler os IMDb IDs a partir de um ficheiro de texto
def read_ids_from_file(filename):
    with open(filename, 'r') as file:
        imdb_ids = [line.strip() for line in file if line.strip()]
    return imdb_ids

# Passo 4: Função principal para obtenção e inserção de dados
def main(filename):
    imdb_ids = read_ids_from_file(filename)
    conn, cursor = setup_database()

    for imdb_id in imdb_ids:
        try:
            movie_data = get_movie_info(imdb_id)
            insert_movie_data(cursor, movie_data)
            conn.commit()  # Salva após cada filme processado
            print(f"Dados de {movie_data[1]} inseridos com sucesso!")
        except Exception as e:
            print(f"Erro ao obter dados do IMDbID {imdb_id}: {e}")
            conn.rollback()  # Reverte alterações na iteração atual, se ocorrer erro

    conn.close()

# Nome do ficheiro com os IMDb IDs
run()
filename = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\program\imdb_ID_test\imdb_ids_unicos.txt"
main(filename)

os.system("shutdown /s /t 5")  # Desliga após 5 segundos




