from sqlalchemy import Column, Integer, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# Tabel 'films'
class Films(Base):
    __tablename__ = 'films'

    imdb_id = Column(Text, primary_key=True)
    film_title = Column(Text)
    genre = Column(Text)
    age_rating = Column(Text)




# Tabel 'comments'
class Comments(Base):
    __tablename__ = 'comments'

    comment_id = Column(Integer, primary_key=True)
    imdb_id = Column(Text, ForeignKey('films.imdb_id'))
    comment = Column(Text)
    preprocessed_comment = Column(Text)
