# GUIDE

# installation:

To run this project, we strongly recommend that you use Conda-Mamba to manage the python environment. 

**NOTE**: We used RoboStack with ROS2-Humble for this project.

Create a new env and install the dependencies:

```
mamba env create -f environment.yml
conda activate propif_ros2
```

To download and install lightglue ensure that the terminal env is pan_seg_3d (in any folder you like):
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

To download and install [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) (HLoc), type following command in any folder you like:

```
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

**NOTE**: Since HLoc is using exactly pycolmap=0.6.0, we recommand you re-install pycolmap=0.6.0:

```
pip install --force-reinstall pycolmap==0.6.0
```

To download and install [CuRobo](https://curobo.org/), type following command in any folder you like:

```
sudo apt install git-lfs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation # This will take 20 minutes to install
```

To download and install [Simulation and Control](https://github.com/VModugno/simulation_and_control/tree/dce4c1da3c28a636a5723bdea38a9de8b1224559), type following command in any folder you like:

```
git clone https://github.com/VModugno/simulation_and_control.git
cd simulation_and_control
pip install .
```

# execute

After installing all the dependencies above, we can start executing the project.

We provide a launch file for you to launch control node, perception node and simulation node together.

Cd to ProPif directory and type following commands to startup a pybullet simulation for this project and launch all 3 nodes.

```
cd propif_ros2
colcon build
source install/setup.bash # change to setup.zsh if you are using zsh
ros2 launch propif_bringup system.launch.py
```

You can also launch 3 nodes seperately:

```
cd propif_ros2
colcon build
```

Now start up 3 terminals, terminal A, B and C.

Inside Terminal A, toi start simulation node. Cd to ProPif/propif_ros2, then:

```
source install/setup.bash # change to setup.zsh if you are using zsh
ros2 launch propif_simulation simulation.launch.py 
```

Inside Terminal B, to start control node. Cd to ProPif/propif_ros2, then:

```
source install/setup.bash # change to setup.zsh if you are using zsh
ros2 launch propif_control control.launch.py 
```

Inside Terminal C, to start perception node. Cd to ProPif/propif_ros2, then:

```
source install/setup.bash # change to setup.zsh if you are using zsh
ros2 launch propif_perception perception.launch.py 
```