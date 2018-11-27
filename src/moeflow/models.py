import enum
import os
import pathlib
import shutil

from appdirs import user_data_dir
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Table, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy_utils import ColorType, ChoiceType
from sqlalchemy_utils.models import Timestamp

from moeflow.util import sha256_checksum


DATA_DIR = user_data_dir('MoeFlow', 'Iskandar Setiadi')
IMAGE_DIR = os.path.join(DATA_DIR, 'image')


Base = declarative_base()
checksum_tags = Table(
    'checksum_tags', Base.metadata,
    Column('checksum_id', ForeignKey('checksum.id'), primary_key=True),
    Column('tag_id', ForeignKey('tag.id'), primary_key=True)
)
checksum_faces = Table(
    'checksum_faces', Base.metadata,
    Column('checksum_id', ForeignKey('checksum.id'), primary_key=True),
    Column('face_id', ForeignKey('face.id'), primary_key=True)
)
face_color_tags = Table(
    'face_color_tags', Base.metadata,
    Column('face_id', ForeignKey('face.id'), primary_key=True),
    Column('color_tag_id', ForeignKey('color_tag.id'), primary_key=True)
)
face_tags = Table(
    'face_tags', Base.metadata,
    Column('face_id', ForeignKey('face.id'), primary_key=True),
    Column('tag_id', ForeignKey('tag.id'), primary_key=True)
)
face_prediction_tags = Table(
    'face_prediction_tags', Base.metadata,
    Column('face_prediction_id', ForeignKey('face_prediction.id'), primary_key=True),
    Column('tag_id', ForeignKey('tag.id'), primary_key=True)
)


class BaseModel(Base, Timestamp):  # type: ignore
    __abstract__ = True
    id = Column(Integer, primary_key=True)


class Checksum(BaseModel):
    __tablename__ = 'checksum'
    value = Column(String, unique=True)
    ext = Column(String)
    trash = Column(Boolean, default=False)
    width = Column(Integer)
    height = Column(Integer)
    tags = relationship('Tag', secondary=checksum_tags, backref='checksums')


class Face(BaseModel):
    __tablename__ = 'face'
    checksum_id = Column(Integer, ForeignKey('checksum.id'))
    checksum = relationship("Checksum", backref="faces", foreign_keys=checksum_id)
    resized_checksum_id = Column(Integer, ForeignKey('checksum.id'))
    resized_checksum = relationship(
        "Checksum", backref="face_models", foreign_keys=resized_checksum_id)
    method = Column(String)
    pos_x = Column(Integer)
    pos_y = Column(Integer)
    pos_w = Column(Integer)
    pos_h = Column(Integer)
    colors = relationship('ColorTag', secondary=face_color_tags, backref='faces')
    tags = relationship('Tag', secondary=face_tags, backref='faces')


class FacePrediction(BaseModel):
    __tablename__ = 'face_prediction'
    face_id = Column(Integer, ForeignKey('face.id'))
    face = relationship("Face", backref="predictions", foreign_keys=face_id)
    method = Column(String)
    confidence = Column(Float)


class FaceComparisonStatus(enum.Enum):
    unknown = 0
    valid = 1
    invalid = 2


class FaceComparison(BaseModel):
    __tablename__ = 'face_comparison'
    left_operand_id = Column(Integer, ForeignKey('face.id'))
    left_operand = relationship(
        'Face', backref='left_face_comparisons', foreign_keys=left_operand_id)
    right_operand_id = Column(Integer, ForeignKey('face.id'))
    right_operand = relationship(
        'Face', backref='right_face_comparisons', foreign_keys=right_operand_id)
    result = Column(Boolean, default=False)
    status = Column(ChoiceType(FaceComparisonStatus, impl=Integer()))
    method = Column(String)


class Tag(BaseModel):
    __tablename__ = 'tag'
    id = Column(Integer, primary_key=True)
    value = Column(String)
    namespace_id = Column(Integer, ForeignKey('namespace.id'))
    namespace = relationship(
        "Namespace", backref="tags", foreign_keys=namespace_id)
    parents = relationship("Tag", backref=backref('children', remote_side=[id]))
    sibling_id = Column(Integer, ForeignKey('tag.id'))
    sibling = relationship("Tag", backref="siblings", foreign_keys=sibling_id, remote_side=[id])


class Namespace(BaseModel):
    __tablename__ = 'namespace'
    value = Column(String)


class ColorTag(BaseModel):
    __tablename__ = 'color_tag'
    value = Column(String)
    color_value = Column(ColorType)


def get_or_create(session, model, **kwargs):
    """Creates an object or returns the object if exists."""
    instance = session.query(model).filter_by(**kwargs).first()
    created = False
    if not instance:
        instance = model(**kwargs)
        session.add(instance)
        created = True
    return instance, created


def add_image(db_session, filename, pil_image=None, image_dir=None):
    """add image to database."""
    checksum = sha256_checksum(filename)
    c_model, created = get_or_create(
        db_session, Checksum, value=checksum)
    if not created:
        raise NotImplementedError
    c_model.ext = pil_image.format.lower()
    c_model.width, c_model.height = pil_image.size
    c_model.trash = False
    if image_dir is not None:
        new_filename =  \
            os.path.join(
                image_dir, checksum[:2], '{}.{}'.format(c_model.value, c_model.ext))
        pathlib.Path(os.path.dirname(new_filename)).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, new_filename)
            
    db_session.add(c_model)
    return c_model, created
