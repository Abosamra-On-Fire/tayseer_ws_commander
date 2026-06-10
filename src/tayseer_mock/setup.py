from setuptools import setup

package_name = 'tayseer_mock'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Mock nodes for Tayseer testing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mock_action_servers = tayseer_mock.mock_action_servers:main',
            'mock_perception = tayseer_mock.mock_perception:main',
        ],
    },
)