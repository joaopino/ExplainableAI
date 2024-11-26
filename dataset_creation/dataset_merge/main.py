import sqlite3
import os

# Caminhos para os arquivos de banco de dados
train_db_path = "imdb_filmes_train.db"
test_db_path = "imdb_filmes_test.db"
output_db_path = "all_films.db"

# Remover o arquivo de saída, caso já exista
if os.path.exists(output_db_path):
    os.remove(output_db_path)

# Criar conexão para a nova base de dados
output_conn = sqlite3.connect(output_db_path)
output_cursor = output_conn.cursor()


# Função para copiar tabelas de uma base de dados para outra
def copy_table(source_db_path, table_name, output_conn):
    source_conn = sqlite3.connect(source_db_path)
    source_cursor = source_conn.cursor()

    # Obter a estrutura da tabela
    source_cursor.execute(f"PRAGMA table_info({table_name})")
    table_info = source_cursor.fetchall()

    # Criar a tabela na base de dados de destino, se não existir
    column_definitions = ", ".join([f"{col[1]} {col[2]}" for col in table_info])
    output_cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})")

    # Transferir os dados
    source_cursor.execute(f"SELECT * FROM {table_name}")
    rows = source_cursor.fetchall()
    placeholders = ", ".join(["?"] * len(table_info))
    output_cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)

    # Fechar a conexão com a base de dados de origem
    source_conn.close()


# Copiar tabelas da base de dados "train"
copy_table(train_db_path, "comments", output_conn)
copy_table(train_db_path, "films", output_conn)

# Copiar tabelas da base de dados "test"
copy_table(test_db_path, "comments", output_conn)
copy_table(test_db_path, "films", output_conn)

# Fechar a conexão com a base de dados de saída
output_conn.commit()
output_conn.close()

print(f"Dados combinados foram salvos em {output_db_path}.")

