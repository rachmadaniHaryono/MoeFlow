from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import ColorType
from sqlalchemy_utils.models import Timestamp


Base = declarative_base()
checksum_tags = Table('checksum_tags', Base.metadata,
    Column('checksum_id', ForeignKey('checksum.id'), primary_key=True),
    Column('tag_id', ForeignKey('checksum.id'), primary_key=True)
)


class Checksum(Base, Timestamp):
    id = Column(Integer, primary_key=True)
    value = Column(String, unique=True)
    ext = Column(String)
    trash = Column(Boolean, default=True)
    width = Column(Integer)
    height = Column(Integer)
    tags = Tag
    #  faces = Face


class Face(Base, Timestamp):
    id = Column(Integer, primary_key=True)
    checksum_id
    checksum
    resized_checksum_id
    resized_checksum
    method
    pos_x = Column(Integer)
    pos_y = Column(Integer)
    pos_w = Column(Integer)
    pos_h = Column(Integer)
    colors = ColorTag
    tags = Tag
    prediction = FacePrediction


class FacePrediction(Base, Timestamp):
    face_id
    face
    method
    tag
    confidence



class Tag(Base):
    value = Column(String)
    namespace_id
    namespace
    parents
    sibling


class Namespace(Base):
    value = Column(String)


class ColorTag(Base):
    value = Column(String)
    color_value = Column(ColorType)
