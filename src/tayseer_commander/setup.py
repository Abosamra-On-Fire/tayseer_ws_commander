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
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'google-generativeai', 'python-dotenv'],
    zip_safe=True,
    maintainer='kareem abosmara',
    maintainer_email='kareem.abosamra02@eng-st.cu.edu.eg',
    description='Commander node for Tayseer robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'commander = tayseer_commander.commander_node:main',
            'world_model = tayseer_commander.world_model:main',
        ],
    },
)