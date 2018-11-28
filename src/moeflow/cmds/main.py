#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import pathlib
import tempfile

import animeface
import click
import cv2
import magic
import PIL.Image
import tensorflow as tf
from sanic import Sanic, response
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from moeflow import models
from moeflow.models import IMAGE_DIR
from moeflow.classify import classify_resized_face
from moeflow.jinja2_env import render
from moeflow.util import (
    get_hex_value,
    get_resized_face_temp_file,
)


app = Sanic(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(dir_path, '..', 'static')
app.static('/static', static_path)
app.static('/i', IMAGE_DIR, name='static_image')
pathlib.Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

ALLOWED_MIMETYPE = ['image/jpeg', 'image/png']


def get_faces(img, db_session=None, c_model=None):
    detector_name = 'python_animeface'
    create_a_new_face_model = False
    af_faces = animeface.detect(img)
    for ff in af_faces:
        face_model = None
        pos = vars(ff.face.pos)  # NOQA
        color_tags = {
            key: value.color for key, value in vars(ff).items()
            if hasattr(value, 'color')}
        face_dict = {
            'detector': detector_name,
            'pos': ff.face.pos,
            'likelihood': ff.likelihood,
            'colors': color_tags,
        }
        if db_session and c_model and c_model.faces:
            # search any existing faces
            for ff_model in c_model.faces:
                same_coordinate = [x for x in pos.values()] == list(ff_model.pos())
                if same_coordinate and ff_model.method == detector_name:
                    face_model = ff_model
                    break
            #  update or create face_model
            if face_model:
                face_model.likelihood = ff.likelihood
                # update color tags on face
                # by removing existing color vlaue
                c_tag_list = []
                for color_m in face_model.color_tags:
                    if color_m.value not in color_tags.values():
                        c_tag_list.append(color_m)
                for key, value in color_tags.items():
                    color_m, _ = models.get_or_create(
                        db_session, models.ColorTag,
                        value=key, color_value=get_hex_value(value.r, value.g, value.b)
                    )
                    db_session.add(color_m)
                    c_tag_list.append(color_m)
                db_session.add(face_model)
                face_model.color_tags = c_tag_list
            else:
                create_a_new_face_model = True
        elif db_session and c_model:
            create_a_new_face_model = True
        if create_a_new_face_model:
            face_model = models.Face(
                checksum=c_model, method=detector_name, likelihood=ff.likelihood)
            for key, value in pos.items():
                setattr(face_model, 'pos_{}'.format(key), value)
            for key, value in color_tags.items():
                color_m, _ = models.get_or_create(
                    db_session, models.ColorTag,
                    value=key, color_value=get_hex_value(value.r, value.g, value.b)
                )
                face_model.color_tags.append(color_m)
                db_session.add(color_m)
            db_session.add(face_model)
        yield [face_dict, face_model]


def predict(filename, config=None, db_session=None):
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
    if db_session:
        db_session.commit()
    # prepare graph and label_lines instance
    if not res['config']['graph'] and not res['config']['label_lines']:
        res['config']['graph'], res['config']['label_lines'] = \
            get_graph_and_label_lines(
                res['config']['model_path'], res['config']['label_path'])
    # predict each face
    label_lines = res['config']['label_lines']
    graph = res['config']['graph']

    if db_session:
        character_nmm, _ = models.get_or_create(
            db_session, models.Namespace, value='character')
    else:
        character_nmm = None
    for idx, (face, face_model) in enumerate(res['faces']):
        resized_path = get_resized_face_temp_file(face, cv2_img)
        res['faces'][idx][0]['resized_path'] = resized_path
        predictions = classify_resized_face(resized_path, label_lines, graph)
        res['faces'][idx][0]['predictions'] = [
            {'method': predict_method_name, 'value': pp[0], 'confidence': pp[1]}
            for pp in predictions
        ]
        if face_model:
            for pp in predictions:
                tag_m, _ = models.get_or_create(
                    db_session, models.Tag, value=pp[0], namespace=character_nmm)
                fp_m, _ = models.get_or_create(
                    db_session, models.FacePrediction,
                    face=face_model, tag=tag_m, method=predict_method_name,
                )
                fp_m.confidence = pp[1]
                pil_r_img = PIL.Image.open(resized_path)
                r_img_model, _ = models.add_image(
                    db_session, resized_path, pil_r_img, IMAGE_DIR)
                face_model.resized_checksum = r_img_model
                db_session.add(r_img_model)
                db_session.add(fp_m)
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
        suffix = '.jpeg' if mime_type == 'image/jpeg' else '.png'  # NOQA
        with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as input_img:  # NOQA
            with open(input_img.name, 'wb') as f:
                f.write(uploaded_image.body)
            db_session = scoped_session(sessionmaker(bind=app.engine))
            config = {'graph': app.graph, 'label_lines': app.label_lines}
            results = predict(input_img.name, db_session=db_session, config=config)
            db_session.commit()
            c_model = results['c_model']
            return response.html(
                render("main.html", c_model=c_model, url_for=app.url_for))
    return response.html(render("main.html"))


@app.route('/face')
async def face_list(request):
    db_session = scoped_session(sessionmaker(bind=app.engine))
    c_models = db_session.query(models.Checksum).filter(models.Checksum.face_models.any()).all()  # NOQA
    return response.html(render('face_list.html', url_for=app.url_for, c_models=c_models))


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

    # init db
    db_uri = 'sqlite:///{}'.format(os.path.join(models.DATA_DIR, 'moeflow.db'))
    engine = create_engine(db_uri)
    models.Base.metadata.create_all(engine)
    app.engine = engine


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
