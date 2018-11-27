# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

requires = [
    'aiofiles==0.3.2',
    'animeface',
    'appdirs==1.4.3',
    'attrs==17.4.0',
    'Click>=7.0',
    'colour==0.1.5',
    'httptools==0.0.9',
    'jinja2==2.10',
    'MarkupSafe==1.0',
    'numpy==1.13.3',
    'opencv-python==3.4.3.18',
    'python-magic==0.4.13',
    'sanic==0.8.3',
    'SQLAlchemy-Utils>=0.33.8',
    'SQLAlchemy>=1.2.14',
    'ujson==1.35',
    'uvloop==0.8.1',
    'websockets>=5.0.1',
]

console_scripts = [
    'moeflow = moeflow.cmds.main:main',
]

setup(
    name='MoeFlow',
    version='0.0.1',
    author='Iskandar Setiadi',
    author_email='iskandarsetiadi@gmail.com',
    url='https://github.com/freedomofkeima/MoeFlow',
    description='Anime characters recognition website, powered by TensorFlow',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={
        '': 'src'
    },
    install_requires=requires,
    #  NOTE: for dependency_links
    #  https://github.com/pypa/pip/issues/3610#issuecomment-356687173
    dependency_links=['http://github.com/nya3jp/python-animeface/tarball/master#egg=animeface-1.1.0'],  # NOQA
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Environment :: Web Environment"
    ],
    entry_points={'console_scripts': console_scripts},
    extras_require={
        'tests': [
            'pytest>=4.0.0', 'pytest-cov', 'pytest-sugar',
            'pytest-asyncio>=0.9.0',  'pytest-flake8>=1.0.2',
        ],
        'patchelf_wrapper': ['patchelf-wrapper==1.0.4', ],
        'tensorflow': ['tensorflow==1.4.0', ],
    },
    zip_safe=False
)
