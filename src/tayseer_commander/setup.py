from setuptools import setup
import os
from glob import glob

package_name = 'tayseer_commander'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        # This is how your PROMPT_PATH gets installed
        (os.path.join('share', package_name, 'config'),
            glob('config/*.txt')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='LLM-based robot commander',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'world_model = tayseer_commander.world_model:main',
            'commander  = tayseer_commander.commander_node:main',
        ],
    },
)