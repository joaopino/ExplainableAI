from sqlalchemy.orm import sessionmaker
from data_preprocessing.database_setup.config import engine
from data_preprocessing.database_setup.models import Comments, Films
import pandas as pd

# Configuração da sessão SQLAlchemy
Session = sessionmaker(bind=engine)
session = Session()

def load_data():
    """
    Carrega os comentários pré-processados e as faixas etárias associadas a cada filme, criando um DataFrame.

    Returns:
        pd.DataFrame: Contém os comentários pré-processados e os rótulos de faixa etária.
    """
    # Query para juntar os comentários pré-processados e as faixas etárias dos filmes
    query = session.query(Comments.preprocessed_comment, Films.age_rating).join(
        Films, Comments.imdb_id == Films.imdb_id
    ).filter(Comments.preprocessed_comment.isnot(None))

    # Converter os dados para um DataFrame
    data = pd.DataFrame(query.all(), columns=["preprocessed_comment", "age_rating"])

    # Mapear as faixas etárias para rótulos numéricos
    age_rating_mapping = {
        "Crianças": 0,
        "Crianças mais velhas": 1,
        "Adolescentes": 2,
        "Adultos": 3
    }
    data["age_rating"] = data["age_rating"].map(age_rating_mapping)

    return data
if __name__ == "__main__":
    data = load_data()
    print(len(data))
