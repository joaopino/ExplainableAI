import sqlite3
import os

# Conectar à base de dados existente com a tabela filmes já populada
def setup_database():
    conn = sqlite3.connect("imdb_filmes.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comentarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            imdb_id TEXT,
            comentario TEXT,
            FOREIGN KEY(imdb_id) REFERENCES filmes(imdb_id)
        )
    ''')
    conn.commit()
    return conn, cursor

# Extrair os IMDb IDs de um ficheiro de URLs
def get_imdb_ids_from_file(file_path):
    imdb_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            url = line.strip()
            imdb_id = url.split("/title/")[1].split("/usercomments")[0]
            imdb_ids.append(imdb_id)
    return imdb_ids

# Inserir comentário associado ao IMDb ID na tabela comentarios
def insert_comment_data(cursor, imdb_id, comment):
    # Verificar se o IMDb ID já existe na tabela filmes
    cursor.execute('SELECT 1 FROM filmes WHERE imdb_id = ?', (imdb_id,))
    if cursor.fetchone():
        # Inserir o comentário se o IMDb ID existe
        cursor.execute('INSERT INTO comentarios (imdb_id, comentario) VALUES (?, ?)', (imdb_id, comment))
    else:
        print(f"IMDb ID {imdb_id} não encontrado na tabela filmes.")

# Processar o ficheiro de URLs e a pasta de comentários
def main(urls_file, comentarios_folder):
    conn, cursor = setup_database()

    imdb_ids = get_imdb_ids_from_file(urls_file)
    comentarios_files = sorted(os.listdir(comentarios_folder))

    if len(imdb_ids) != len(comentarios_files):
        print("Erro: O número de IMDb IDs e de comentários não corresponde.")
        return

    for imdb_id, comentario_file in zip(imdb_ids, comentarios_files):
        try:
            with open(os.path.join(comentarios_folder, comentario_file), 'r') as f:
                comentario = f.read().strip()

            insert_comment_data(cursor, imdb_id, comentario)
            conn.commit()  # Guardar cada inserção

        except Exception as e:
            print(f"Erro ao processar {comentario_file}: {e}")
            conn.rollback()  # Reverter alterações se houver erro na iteração atual

    conn.close()


def run():
    # Caminhos para o ficheiro de URLs e a pasta de comentários
    urls_file = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\Dataset\aclImdb\test\urls_neg.txt"
    comentarios_folder = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\Dataset\aclImdb\test\neg"
    main(urls_file, comentarios_folder)

    urls_file = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\Dataset\aclImdb\test\urls_pos.txt"
    comentarios_folder = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\Dataset\aclImdb\test\pos"
    main(urls_file, comentarios_folder)



