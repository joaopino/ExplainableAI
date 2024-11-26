# Lê os URLs dos ficheiros de entrada e extrai IDs únicos
def extrair_ids_unicos(ficheiros_entrada, ficheiro_saida):
    imdb_ids = set()  # Usa um set para armazenar IDs únicos

    for ficheiro in ficheiros_entrada:
        with open(ficheiro, 'r') as f:
            for linha in f:
                # Extrai o IMDb ID usando slicing
                if "/title/tt" in linha:
                    inicio = linha.index("/title/tt") + len("/title/")
                    fim = inicio + 9  # Os IDs têm sempre 9 caracteres
                    imdb_id = linha[inicio:fim]

                    # Adiciona o ID ao set se ainda não existir
                    imdb_ids.add(imdb_id)

    # Escreve os IDs únicos no ficheiro de saída
    with open(ficheiro_saida, 'w') as f_saida:
        for imdb_id in sorted(imdb_ids):  # Ordena antes de escrever, opcional
            f_saida.write(imdb_id + '\n')

# Define os ficheiros de entrada e o ficheiro de saída
ficheiros_entrada = [r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\Dataset\aclImdb\test\urls_neg.txt", r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\Dataset\aclImdb\test\urls_pos.txt"]
ficheiro_saida = r"C:\Users\João Fonseca Antunes\OneDrive\Ambiente de Trabalho\Mestrado 1 ano\Inteligencia Artificial Centrada no Humano\projeto\program\imdb_ID_test\imdb_ids_unicos.txt"

# Executa a função
extrair_ids_unicos(ficheiros_entrada, ficheiro_saida)
