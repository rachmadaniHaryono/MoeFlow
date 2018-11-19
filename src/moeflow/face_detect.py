# -*- coding: utf-8 -*-
import logging
import os
import subprocess
import tempfile

import animeface
import PIL.Image


def run_face_detection(input_image_path):
    """
    Receives input image path
    Return list of path (detected faces in /tmp directory)
    """
    args = [
        'ruby',
        'detect.rb',
        input_image_path,
        '/tmp'
    ]
    results = []
    # Execute
    try:
        output = subprocess.check_output(  # NOQA
            args,
            shell=False,
            timeout=30
        )
    except subprocess.CalledProcessError as e:
        logging.debug("{}:{}".format(type(e), e))
        logging.debug("Use python animeface.")
        return run_face_detection_with_python_animeface(input_image_path)
    except Exception:
        logging.exception("Face detection failed!")
        return []
    input_name_base = os.path.basename(input_image_path)
    input_name_without_ext = os.path.splitext(input_name_base)[0]
    for filename in os.listdir('/tmp'):
        if filename.startswith(input_name_without_ext + '_out'):
            results.append("/tmp/{}".format(filename))
    return results


def run_face_detection_with_python_animeface(input_image_path):
    res = []
    im = PIL.Image.open(input_image_path)
    faces = animeface.detect(im)
    for ff in faces:
        with tempfile.NamedTemporaryFile(delete=False) as temp_ff:
            temp_im = im.crop((
                ff.face.pos.x,
                ff.face.pos.y,
                ff.face.pos.x + ff.face.pos.width,
                ff.face.pos.y + ff.face.pos.height,
            ))
            name = '{}.jpg'.format(temp_ff.name)
            temp_im.save(name)
            res.append(name)
    return res
