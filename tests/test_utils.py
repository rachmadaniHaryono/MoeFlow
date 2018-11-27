import cv2
import PIL.Image


def test_get_resized_face_temp_file():
    from moeflow.cmds.main import get_faces
    from moeflow.util import get_resized_face_temp_file
    filename = 'screenshots/altered_2_characters.png'
    img = PIL.Image.open(filename)
    faces = list(get_faces(img))
    face0 = faces[0][0]
    cv2_img = cv2.imread(filename)
    res = get_resized_face_temp_file(face_dict=face0, cv2_img=cv2_img)
    assert res
