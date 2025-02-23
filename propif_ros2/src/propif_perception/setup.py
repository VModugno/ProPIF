# file: propif_ros2/src/propif_perception/setup.py
import os
from glob import glob
from setuptools import setup

package_name = 'propif_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],  # 如果你在 propif_perception/propif_perception/ 有 __init__.py
    package_dir={'': '.'},
    data_files=[
        # 安装 package.xml
        (os.path.join('share', package_name), ['package.xml']),
        # 如果你有 launch 文件夹，需要安装它们以便 'ros2 launch' 能找到
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # 如果你有 models 文件夹需要一起安装，也可加：
        # (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],  # Python依赖，比如 numpy, opencv-python, torch, etc. 可自行补充
    zip_safe=True,
    maintainer='YourName',
    maintainer_email='minchuan.yang.24@ucl.ac.uk',
    description='ROS2 Python package for perception, PyBullet sim, mechanical arm, etc.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 下面只是举例，根据你实际脚本命名
            # 让 Pan3D.py 成为一个可执行节点 (需在 Pan3D.py 有 main() 函数)
            'pan3d_node = propif_perception.scripts.Pan3d:main',

            # 如果你有 PyBullet 仿真脚本:
            'pybullet_sim = propif_perception.scripts.pybullet_sim:main',

            # 机械臂操作节点:
            'manipulator_node = propif_perception.scripts.manipulator:main',
        ],
    },
)
