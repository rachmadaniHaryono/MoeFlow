#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import pathlib
import shutil
import tempfile

import animeface
import click
import cv2
import magic
import PIL.Image
import tensorflow as tf
from sanic import Sanic, response

from moeflow import models
from moeflow.models import IMAGE_DIR
from moeflow.classify import classify_resized_face
from moeflow.face_detect import run_face_detection
from moeflow.jinja2_env import render
from moeflow.util import (
    cleanup_image_cache,
    resize_faces,
    resize_large_image,
    sha256_checksum,
)


app = Sanic(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(dir_path, '..', 'static')
app.static('/static', static_path)
app.static('/i', IMAGE_DIR)
pathlib.Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

ALLOWED_MIMETYPE = ['image/jpeg', 'image/png']


def get_faces(img, db_session=None, c_model=None):
    detector_name = 'python_animeface'
    valid_color_attrs = ['skin', 'hair', 'left_eye', 'right_eye']
    af_faces = animeface.detect(img)
    for idx, ff in enumerate(af_faces):
        face_model = None
        face_dict = {
            'idx': idx,
            'detector': detector_name,
            'pos': ff.face.pos,
            'likelihood': ff.likelihood,
            'colors': {
                key: value.color for key, value in vars(ff).items()
                if key in valid_color_attrs}
        }
        if db_session and c_model:
            raise NotImplementedError
        yield [face_dict, face_model]


def predict(filename, config=None, db_session=None):
    width, height = 96, 96
    predict_method_name = 'moeflow'
    pil_img = PIL.Image.open(filename)
    cv2_img = cv2.imread(filename)
    created = True
    c_model = None

    if db_session is not None:
        c_model, created = models.add_image(db_session, filename, pil_img, IMAGE_DIR)
        db_session.commit()

    res = {
        'filename': filename,
        'pil_img': pil_img,
        'cv2_img': cv2_img,
        'c_model': c_model,
        'faces': [],
        'config': {'graph': None, 'label_lines': None}}
    if config is not None:
        res['config'].update(config)
    # detect face with python anime_face
    res['faces'] = list(get_faces(pil_img, db_session, c_model))
    # prepare graph and label_lines instance
    if not res['config']['graph'] and not res['config']['label_lines']:
        res['config']['graph'], res['config']['label_lines'] = \
            get_graph_and_label_lines(
                res['config']['model_path'], res['config']['label_path'])
    # predict each face
    label_lines = res['config']['label_lines']
    graph = res['config']['graph']
    for idx, (face, face_model) in enumerate(res['faces']):
        pos = face['pos']
        crop_img = cv2_img[pos.y:pos.y+pos.height, pos.x:pos.x+pos.width]
        resized_img = cv2.resize(
            crop_img,
            (width, height),
            interpolation=cv2.INTER_AREA
        )
        resized_path = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_ff:
            resized_path = temp_ff.name + '.jpg'
            cv2.imwrite(temp_ff.name + '.jpg', resized_img)
        res['faces'][idx][0]['resized_path'] = temp_ff.name
        predictions = classify_resized_face(resized_path, label_lines, graph)
        res['faces'][idx][0]['predictions'] = [
            {'method': predict_method_name, 'value': x[0], 'confidence': x[1]}
            for x in predictions]
    return res


def get_graph_and_label_lines(model_path, label_path):
    label_lines = [
        line.strip() for line in tf.gfile.GFile(label_path)
    ]
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph, label_lines


@app.route("/", methods=['GET', 'POST'])
async def main_app(request):
    if request.method == "POST":
        uploaded_image = request.files.get('uploaded_image')
        mime_type = magic.from_buffer(uploaded_image.body, mime=True)
        if mime_type not in ALLOWED_MIMETYPE:
            return response.html(render("main.html"))
        # Scale down input image to ~800 px
        image = resize_large_image(uploaded_image.body)
        with tempfile.NamedTemporaryFile(mode="wb", suffix='.jpg') as input_jpg:
            filename = input_jpg.name
            logging.info("Input file is created at {}".format(filename))
            cv2.imwrite(filename, image)
            # Copy to image directory
            csm = sha256_checksum(filename)
            ori_name = csm + ".jpg"
            ori_path = os.path.join(static_path, 'images', ori_name[:2], ori_name)
            pathlib.Path(os.path.dirname(ori_path)).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filename, ori_path)
            results = []
            # Run face detection with animeface-2009
            detected_faces = run_face_detection(filename)
            # This operation will rewrite detected faces to 96 x 96 px
            resize_faces(detected_faces)
            # Classify with TensorFlow
            if not detected_faces:  # Use overall image as default
                detected_faces = [filename]
            for face in detected_faces:
                predictions = classify_resized_face(
                    face,
                    app.label_lines,
                    app.graph
                )
                face_name = sha256_checksum(face) + ".jpg"
                face_name_path = os.path.join(
                    static_path, 'images', face_name[:2], face_name)
                pathlib.Path(
                    os.path.dirname(face_name_path)
                ).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(face, face_name_path)
                results.append({
                    "image_name": face_name,
                    "prediction": predictions
                })
                logging.info(predictions)
            # Cleanup
            cleanup_image_cache(os.path.join(static_path, 'images'))
            for faces in detected_faces:
                if faces != filename:
                    os.remove(faces)
        return response.html(
            render(
                "main.html",
                ori_name=ori_name,
                results=results
            )
        )
    return response.html(render("main.html"))


@app.route("/hello_world")
async def hello_world(request):
    return response.text("Hello world!")


@app.listener('before_server_start')
async def initialize(app, loop):
    moeflow_path = os.environ.get('MOEFLOW_MODEL_PATH', '')
    label_path = os.path.join(os.sep, moeflow_path, "output_labels_2.txt")
    model_path = os.path.join(os.sep, moeflow_path, "output_graph_2.pb")
    app.graph, app.label_lines = get_graph_and_label_lines(model_path, label_path)
    logging.info("MoeFlow model is now initialized!")


@click.command()
@click.option('--host', help='Host.')
@click.option('--port', help='Port number.', type=int)
@click.option('--debug', help='Run debugging.', is_flag=True)
def main(host, port, debug):
    # Set logger
    logging.basicConfig(level=logging.DEBUG if not debug else logging.INFO)
    kwargs = {}
    if host:
        kwargs['host'] = host
    if port:
        kwargs['port'] = port
    if debug:
        kwargs['debug'] = debug
    app.run(**kwargs)


if __name__ == '__main__':
    main()
