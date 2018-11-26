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
    assert res['faces'][0]['colors']


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
    with pytest.raises(NotImplementedError):
        predict('screenshots/altered_2_characters.png', config, db_session)
    #  assert res
