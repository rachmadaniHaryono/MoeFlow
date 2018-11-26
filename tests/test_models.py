from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from moeflow import models


def test_init():
    engine = create_engine('sqlite:///:memory:')
    Session = sessionmaker(bind=engine)
    session = Session()
    models.Base.metadata.create_all(engine)
    assert session
