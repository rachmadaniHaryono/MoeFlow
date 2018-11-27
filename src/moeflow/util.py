# -*- coding: utf-8 -*-
import hashlib
import logging
import os
import time

import cv2
import numpy as np

WIDTH_HEIGHT_LIMIT = 1600  # in pixel


def resize_large_image(image_data):
    img_array = np.fromstring(image_data, dtype=np.uint8)
    image = cv2.imdecode(img_array, 1)
    height, width = image.shape[:2]
    logging.info("Height: {}, Width: {}".format(height, width))
    if height > width and height > WIDTH_HEIGHT_LIMIT:
        ratio = float(WIDTH_HEIGHT_LIMIT) / float(height)
        new_width = int((width * ratio) + 0.5)
        return cv2.resize(
            image,
            (new_width, WIDTH_HEIGHT_LIMIT),
            interpolation=cv2.INTER_AREA
        )
    elif width > WIDTH_HEIGHT_LIMIT:
        ratio = float(WIDTH_HEIGHT_LIMIT) / float(width)
        new_height = int((height * ratio) + 0.5)
        return cv2.resize(
            image,
            (WIDTH_HEIGHT_LIMIT, new_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        return image


def resize_faces(image_files, width=96, height=96):
    for image_file in image_files:
        image = cv2.imread(image_file)
        resized_image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(image_file, resized_image)


def cleanup_image_cache(image_dir, expire=3600):  # Expire in 1 hour
    now = time.time()
    for f in os.listdir(image_dir):
        f = os.path.join(image_dir, f)
        if os.stat(f).st_mtime < now - expire:
            if os.path.isfile(f):
                os.remove(f)


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


def get_hex_value(r, g, b):
    def clamp(x):
        return max(0, min(x, 255))

    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))
