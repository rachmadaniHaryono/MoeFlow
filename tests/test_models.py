from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import PIL.Image

from moeflow import models


def test_init():
    engine = create_engine('sqlite:///:memory:')
    Session = sessionmaker(bind=engine)
    session = Session()
    models.Base.metadata.create_all(engine)
    assert session


def test_add_image():
    engine = create_engine('sqlite://')
    models.Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db_session = Session()
    filename = 'screenshots/altered_2_characters.png'
    img = PIL.Image.open(filename)
    c_model1, created = models.add_image(db_session, filename, pil_image=img)
    db_session.commit()
    assert created
    c_model2, created = models.add_image(db_session, filename, pil_image=img)
    db_session.commit()
    assert not created
    assert c_model1.id == c_model2.id
