import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest


MODEL_FOLDER = os.path.join(os.path.dirname(__file__), '..')
LABEL_PATH = os.path.join(MODEL_FOLDER, 'output_labels_2.txt')
MODEL_PATH = os.path.join(MODEL_FOLDER, 'output_graph_2.pb')


@pytest.fixture
def graph_and_label_lines():
    from moeflow.cmds.main import get_graph_and_label_lines
    res = get_graph_and_label_lines(MODEL_PATH, LABEL_PATH)
    return res


def test_get_graph_and_label_lines():
    from moeflow.cmds.main import get_graph_and_label_lines
    res = get_graph_and_label_lines(MODEL_PATH, LABEL_PATH)
    assert res


def test_predict(graph_and_label_lines):
    from moeflow.cmds.main import predict
    graph, label_lines = graph_and_label_lines
    config = {'graph': graph, 'label_lines': label_lines}
    res = predict('screenshots/altered_2_characters.png', config)
    assert res['faces']
    assert res['faces'][0][0]['colors']


def test_predict_with_db(graph_and_label_lines):
    from moeflow.cmds.main import predict
    from moeflow import models
    graph, label_lines = graph_and_label_lines
    config = {'graph': graph, 'label_lines': label_lines}
    # db session
    engine = create_engine('sqlite:///:memory:')

    Session = sessionmaker(bind=engine)
    db_session = Session()
    models.Base.metadata.create_all(engine)
    res = predict('screenshots/altered_2_characters.png', config, db_session)
    assert res
    db_session.commit()
    assert res['c_model'].faces[0].predictions


def test_get_faces():
    import PIL.Image

    from moeflow.cmds.main import get_faces
    img = PIL.Image.open('screenshots/altered_2_characters.png')
    res = list(get_faces(img))
    assert res[0][0]
    assert not res[0][1]
    assert len(res) == 4


def test_get_faces_with_db():
    import PIL.Image

    from moeflow.cmds.main import get_faces
    from moeflow import models
    engine = create_engine('sqlite://')
    models.Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db_session = Session()
    filename = 'screenshots/altered_2_characters.png'
    img = PIL.Image.open(filename)
    c_model, created = models.add_image(db_session, filename, pil_image=img)
    assert created
    res = list(get_faces(img, db_session, c_model))
    assert res[0][0]
    assert len(res) == 4
    assert res[0][1]
    assert res[0][1].color_tags
    db_session.commit()

    #  get the same face with existing db
    c_model2, created = models.add_image(db_session, filename, pil_image=img)
    db_session.commit()
    assert not created
    assert c_model.id == c_model2.id
