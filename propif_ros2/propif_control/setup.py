from setuptools import setup
import os
from glob import glob

package_name = 'propif_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],  # Simplify to match perception setup
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch file
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='steve',
    maintainer_email='steveyang520314@gmail.com',
    description='Robot control package for ProPIF project',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = propif_control.control_node:main',
        ],
    },
)