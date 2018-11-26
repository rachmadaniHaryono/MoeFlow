from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Table, Float
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import ColorType
from sqlalchemy_utils.models import Timestamp


Base = declarative_base()
checksum_tags = Table(
    'checksum_tags', Base.metadata,
    Column('checksum_id', ForeignKey('checksum.id'), primary_key=True),
    Column('tag_id', ForeignKey('checksum.id'), primary_key=True)
)
checksum_faces = Table(
    'checksum_faces', Base.metadata,
    Column('checksum_id', ForeignKey('checksum.id'), primary_key=True),
    Column('face_id', ForeignKey('face.id'), primary_key=True)
)
face_color_tags = Table(
    'face_color_Tags', Base.metadata,
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


class BaseModel(Base, Timestamp):
    __abstract__ = True
    id = Column(Integer, primary_key=True)


class Checksum(BaseModel):
    __tablename__ = 'checksum'
    value = Column(String, unique=True)
    ext = Column(String)
    trash = Column(Boolean, default=False)
    width = Column(Integer)
    height = Column(Integer)
    tags = relationship('Tag', secondary=checksum_tags, back_populates='checksums')


class Face(BaseModel):
    __tablename__ = 'face'
    checksum_id = Column(Integer, ForeignKey('checksum.id'))
    checksum = relationship("Checksum", back_populates="faces", foreign_keys=checksum_id)
    resized_checksum_id = Column(Integer, ForeignKey('checksum.id'))
    resized_checksum = relationship(
        "Checksum", back_populates="faces", foreign_keys=resized_checksum_id)
    method = Column(String)
    pos_x = Column(Integer)
    pos_y = Column(Integer)
    pos_w = Column(Integer)
    pos_h = Column(Integer)
    colors = relationship('ColorTag', secondary=face_color_tags, back_populates='faces')
    tags = relationship('Tag', secondary=face_tags, back_populates='faces')


class FacePrediction(BaseModel):
    __tablename__ = 'face_prediction'
    face_id = Column(Integer, ForeignKey('face.id'))
    face = relationship("Face", back_populates="predictions", foreign_keys=face_id)
    method = Column(String)
    tags = relationship('Tag', secondary=face_tags, back_populates='face_predictions')
    confidence = Column(Float)


class Tag(BaseModel):
    __tablename__ = 'tag'
    value = Column(String)
    namespace_id = Column(Integer, ForeignKey('namespace.id'))
    namespace = relationship(
        "Namespace", back_populates="tags", foreign_keys=namespace_id)
    parents = relationship("Tag", backref=backref('children', remote_side=[id]))
    sibling_id = Column(Integer, ForeignKey('tag.id'))
    sibling = relationship("Tag", back_populates="siblings", foreign_keys=sibling_id)


class Namespace(BaseModel):
    __tablename__ = 'namespace'
    value = Column(String)


class ColorTag(BaseModel):
    __tablename__ = 'color_tag'
    value = Column(String)
    color_value = Column(ColorType)
