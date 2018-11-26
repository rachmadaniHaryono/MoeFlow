def test_predict():
    from moeflow.cmds.main import predict
    config = {
        'label_path': 'models/output_labels_2.txt',
        'model_path': 'models/output_graph_2.pb',
    }
    res = predict('screenshots/altered_2_characters.png', config)
    assert res['faces']
    assert res['faces'][0]['colors']
