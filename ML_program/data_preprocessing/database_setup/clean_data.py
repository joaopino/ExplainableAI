import re
from sqlalchemy.orm import sessionmaker
from ML_program.data_preprocessing.database_setup.config import engine
from ML_program.data_preprocessing.database_setup.models import Films

# Configurar sessão SQLAlchemy
Session = sessionmaker(bind=engine)
session = Session()

# Dicionário de mapeamento das classificações
age_rating_mapping = {
    "Crianças": [
        "United States:G", "United States:GA", "United States:TV-Y",
        "United States:TV-Y7", "United States:Approved", "United States:TV-G",
        "United States:Passed", "United States:E"
    ],
    "Crianças mais velhas": [
        "United States:TV-Y7-FV", "United States:TV-Y7", "United States:PG"
    ],
    "Adolescentes": [
        "United States:PG-13", "United States:13+", "United States:T",
        "United States:TV-PG", "United States:TV-14"
    ],
    "Jovens adultos": [
        "United States:16+", "United States:M", "United States:M/PG"
    ],
    "Adultos": [
        "United States:R", "United States:NC-17", "United States:X",
        "United States:TV-MA", "United States:Unrated"
    ]
}


def categorize_age_rating(age_rating):
    """
    Categoriza a faixa etária com base no mapeamento.

    Args:
        age_rating (str): A classificação etária original.

    Returns:
        str: A categoria correspondente ou 'Desconhecida' se não encontrar.
    """
    for category, ratings in age_rating_mapping.items():
        if any(rating in age_rating for rating in ratings):
            return category
    return "Desconhecida"


def update_age_ratings():
    """
    Atualiza as classificações etárias na base de dados com base nas categorias.
    """
    films = session.query(Films).all()

    for film in films:
        if film.age_rating:
            categorized_rating = categorize_age_rating(film.age_rating)
            film.age_rating = categorized_rating
            print(f"Atualizado {film.film_title} para {categorized_rating}")

    session.commit()


def main():
    """
    Executa a atualização de classificações etárias.
    """
    update_age_ratings()


if __name__ == "__main__":
    main()
