import os


MODEL_FOLDER = os.path.join(os.path.dirname(__file__), '..')
LABEL_PATH = os.path.join(MODEL_FOLDER, 'output_labels_2.txt')
MODEL_PATH = os.path.join(MODEL_FOLDER, 'output_graph_2.pb')


def test_classify_resized_face():
    import cv2
    import PIL.Image

    from moeflow.classify import classify_resized_face
    from moeflow.cmds.main import get_faces, get_graph_and_label_lines
    from moeflow.util import get_resized_face_temp_file
    filename = 'screenshots/altered_2_characters.png'
    img = PIL.Image.open(filename)
    cv2_img = cv2.imread(filename)
    graph, label_lines = get_graph_and_label_lines(MODEL_PATH, LABEL_PATH)
    faces = list(get_faces(img))
    face0 = faces[0][0]
    rf_tfile = get_resized_face_temp_file(face_dict=face0, cv2_img=cv2_img)
    res = classify_resized_face(rf_tfile, label_lines, graph)
    assert res
    assert len(res) == 3
