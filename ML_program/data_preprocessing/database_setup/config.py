from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure the URI of the Data base:
DATABASE_URI = r"sqlite:///ML_program/all_films.db"



# Create the connection engine and the session
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)


def get_session():
    return Session()