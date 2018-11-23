# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

requires = [
    'aiofiles==0.3.2',
    'attrs==17.4.0',
    'httptools==0.0.9',
    'jinja2==2.10',
    'MarkupSafe==1.0',
    'numpy==1.13.3',
    'opencv-python==3.4.3.18',
    'python-magic==0.4.13',
    'sanic==0.6.0',
    'ujson==1.35',
    'uvloop==0.8.1',
    'websockets==4.0.1',
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
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Environment :: Web Environment"
    ],
    entry_points={'console_scripts': console_scripts},
    extras_require={
        'tests': ['pytest', 'pytest-cov', 'pytest-sugar'],
        'patchelf_wrapper': ['patchelf-wrapper==1.0.4'],
        'tensorflow': ['tensorflow==1.4.0'],
        'python_animeface': ['git+git://github.com/nya3jp/python-animeface.git'],
    },
    zip_safe=False
)
