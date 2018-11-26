import os


def test_predict():
    from moeflow.cmds.main import predict
    model_folder = os.path.join(os.path.dirname(__file__), '..')
    config = {
        'label_path': os.path.join(model_folder, 'output_labels_2.txt'),
        'model_path': os.path.join(model_folder, 'output_graph_2.pb'),
    }
    res = predict('screenshots/altered_2_characters.png', config)
    assert res['faces']
    assert res['faces'][0]['colors']
