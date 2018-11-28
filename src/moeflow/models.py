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

    @property
    def filename(self):
        return os.path.join(self.value[:2], '{}.{}'.format(self.value, self.ext))

    @property
    def face_model_color_tags(self):
        res = []
        unique_list = []
        if self.face_models:
            for fm in self.face_models:
                for c_tag in fm.color_tags:
                    unique_set = (c_tag.value, c_tag.color_value.hex_l)
                    if unique_set not in unique_list:
                        res.append(c_tag)
                        unique_list.append(unique_set)
        return res


class Face(BaseModel):
    __tablename__ = 'face'
    checksum_id = Column(Integer, ForeignKey('checksum.id'))
    checksum = relationship("Checksum", backref="faces", foreign_keys=checksum_id)
    resized_checksum_id = Column(Integer, ForeignKey('checksum.id'))
    resized_checksum = relationship(
        "Checksum", backref="face_models", foreign_keys=resized_checksum_id)
    method = Column(String)
    likelihood = Column(Float)
    pos_x = Column(Integer)
    pos_y = Column(Integer)
    pos_width = Column(Integer)
    pos_height = Column(Integer)
    color_tags = relationship('ColorTag', secondary=face_color_tags, backref='faces')
    tags = relationship('Tag', secondary=face_tags, backref='faces')

    def pos(self):
        return (self.pos_x, self.pos_y, self.pos_width, self.pos_height)

    def __repr__(self):
        return 'Face(id={}, checksum={}, method={}, likelihood={}, pos={})'.format(
            self.id, self.checksum.value[:7], self.method, self.likelihood,
            '({}, {};{}, {})'.format(
                self.pos_x, self.pos_y, self.pos_width, self.pos_height)
        )


class FacePrediction(BaseModel):
    __tablename__ = 'face_prediction'
    face_id = Column(Integer, ForeignKey('face.id'))
    face = relationship("Face", backref="predictions", foreign_keys=face_id)
    tag_id = Column(Integer, ForeignKey('tag.id'))
    tag = relationship("Tag", backref="face_predictions", foreign_keys=tag_id)
    method = Column(String)
    confidence = Column(Float)

    def __repr__(self):
        templ = 'FacePrediction(id={}, face_id={}, tag={}, method={}, confidence={})'
        return templ.format(
            self.id, self.face_id, repr(self.tag), self.method, self.confidence)


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
    sibling = relationship(
        "Tag", backref="siblings", foreign_keys=sibling_id, remote_side=[id])

    def __repr__(self):
        return 'Tag(id={} {})'.format(
            self.id,
            '{}:{}'.format(self.namespace.value, self.value)
            if self.namespace else self.value
        )


class Namespace(BaseModel):
    __tablename__ = 'namespace'
    value = Column(String)


class ColorTag(BaseModel):
    __tablename__ = 'color_tag'
    value = Column(String)
    color_value = Column(ColorType)

    def __repr__(self):
        return 'ColorTag(id={}, value={}, color_value={})'.format(
            self.id, self.value, self.color_value)


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
        if pil_image is not None:
            pass
        else:
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
