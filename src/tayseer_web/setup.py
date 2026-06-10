from setuptools import setup
from glob import glob
import os

package_name = 'tayseer_web'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/web', glob('web/*')),
    ],
    install_requires=['setuptools', 'fastapi', 'uvicorn', 'websockets'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Web interface for Tayseer',
    license='MIT',
    entry_points={
        'console_scripts': [
            'web_server = tayseer_web.web_server:main',
        ],
    },
)