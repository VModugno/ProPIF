# Install ultralytics (yolo and sam)

mamba create --name yolo_sam
activate mamba yolo_sam
mamba install pip
pip install ultralytics


# Install semantic-sam
mamba create --name sem_sam  
activate mamba ensem_samv
mamba install pip
pip install torch torchvision torchaudio

search for the right nvidia toolkit version matching you nvidia cuda driver version(to check nvidia toolkit you can run nvcc --version if you want to check nvidia cuda driver you do nvidia-smi) in this webpage: https://anaconda.org/nvidia/cuda-toolkit in my case was:

mammba install nvidia/label/cuda-12.1.0::cuda-toolkit
pip install git+https://github.com/MaureenZOU/detectron2-xyz.git
pip install git+https://github.com/cocodataset/panopticapi.git
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/models/ops
pip install . 
cd ..
git clone https://github.com/UX-Decoder/Semantic-SAM
cd Semantic-SAM
pip install -r requirements.txt

while installing semantic-sam you can encouter several issues

if you encouter an error: AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'? you need to change the in the requirements.txt of semantic-sam the numpy version to 1.26.4 (line 10 fo the requirements.txt file)

if you encounter an error like error: can't find Rust compiler
you need to install rust using this commands

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

to update the terminal enviromental variables do
source $HOME/.cargo/env 

and to test the rsut version you can do
rustc --version

if  while installing pillow you get this error:

The headers or library files could not be found for jpeg,
a required dependency when compiling Pillow from source.
you can fix it by installing:

sudo apt-get install libjpeg-dev

if you encounter this error:
Could not find directory of OpenSSL installation, and this `-sys` crate cannot
proceed without this knowledge. If OpenSSL is installed and this crate had
trouble finding it,  you can set the `OPENSSL_DIR` environment variable for the
compilation process.
      
 Make sure you also have the development packages of openssl installed.
For example, `libssl-dev` on Ubuntu or `openssl-devel` on Fedora.

you can solve it by pointing the openSSL_DIR to your current mamba environment (assuming openssl is installed in your env)

export OPENSSL_DIR=/home/robohikeuser/mambaforge/envs/sem_sam/


if you encouter this error:

warning: variable does not need to be mutable
     --> tokenizers-lib\src\models\unigram\model.rs:265:21
      |
  265 |                 let mut target_node = &mut best_path_ends_at[key_pos];
      |                     ----^^^^^^^^^^^
      |                     |
      |                     help: remove this `mut`
  ...
  error: casting `&T` to `&mut T` is undefined behavior, even if the reference is unused, consider instead using an `UnsafeCell`
     --> tokenizers-lib\src\models\bpe\trainer.rs:526:47
      |
  522 |                     let w = &words[*i] as *const _ as *mut _;
      |                             -------------------------------- casting happened here
  ...
  526 |                         let word: &mut Word = &mut (*w);
      |                                               ^^^^^^^^^
      |
      = note: for more information, visit <https://doc.rust-lang.org/book/ch15-05-interior-mutability.html>
      = note: `#[deny(invalid_reference_casting)]` on by default

warning: `tokenizers` (lib) generated 3 warnings
error: could not compile `tokenizers` (lib) due to the previous error; 3 warnings emitted


this is due to the fact that rust has become more strict. To fix this you can go on the requirements.txt and replace line 13 with

transformers>=4.36

if you encouter this error:

× python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [6 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-7nawt3_d/pathtools_cac140aad1f84826b512b57e3b6465e7/setup.py", line 25, in <module>
          import imp
      ModuleNotFoundError: No module named 'imp'
      [end of output]

it is because the pathtools module (which is called by wandb) is requiring imp which has been deprecated. a way to fix it is to use a more recent wandb version 

"wandb==0.16.6"

if you get this error:

Make Error at CMakeLists.txt:261 (find_package):
        By not providing "FindArrow.cmake" in CMAKE_MODULE_PATH this project has
        asked CMake to find a package configuration file provided by "Arrow", but
        CMake did not find one.
      
        Could not find a package configuration file provided by "Arrow" with any of
        the following names:
      
          ArrowConfig.cmake
          arrow-config.cmake
      
        Add the installation prefix of "Arrow" to CMAKE_PREFIX_PATH or set
        "Arrow_DIR" to a directory containing one of the above files.  If "Arrow"
        provides a separate development package or SDK, be sure it has been
        installed.

if you update the pyarrow dependencies with this it should work

    "pyarrow==16.0.0"


this will complete the installation

#yolo-world installation

https://blog.roboflow.com/what-is-yolo-world/
https://docs.ultralytics.com/models/yolo-world/#set-prompts
https://huggingface.co/spaces/SkalskiP/YOLO-World
https://supervision.roboflow.com/develop/notebooks/zero-shot-object-detection-with-yolo-world/#final-result

git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
pip install torch wheel -q
pip install -e .




#literature:

https://www.sciencedirect.com/science/article/pii/S092427162300223X
https://nihalsid.github.io/panoptic-lifting/
https://github.com/xheon/panoptic-reconstruction





# NEW GUIDE

# installation:

first install the mamba env

```
mamba env create -f environment.yml
conda activate pan_seg_3d
```

to download and install lightglue ensure that the terminal env is pan_seg_3d (in any folder you like):
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```
then the packaged need to be compiled
```
cd to/ros/workspace
catkin build
```



under the mamba env to start the camera and you get this error:

even after refreshing the cache: Failed to load library /home/vale/mambaforge/envs/pan_seg_3d/lib//librealsense2_camera.so. Make sure that you are calling the PLUGINLIB_EXPORT_CLASS macro in the library code, and that names are consistent between this macro and your XML. Error string: Could not load library (Poco exception = libudev.so.0: cannot open shared object file: No such file or directory)

it means that the librealsense2_camera.so is not linking correctly to the libudev.so.0 library (you can check it by doing from terminal ldd librealsense2_camera.so in the folder in which the lib is contained)
this is due to the fact that probably you have libudev.so.1 (a more recent version of libudev installed)
in order to fix go to the path/to/pan_seg_3d/lib/ an in a terminal type:

```
 sudo ln -s libudev.so.1 libudev.so.0
```

to create a symbolic link to libudev.so.1 inside the pan_seg_3d/lib/ folder. This shoudl fix the issue



# execute

Then open up terminals A, B


terminal A
```
eval "$(/home/vale/mambaforge/bin/conda shell.bash hook)" 
conda activate pan_seg_3d 
cd  to/ros/workspace/cotaining/packages
source ./devel/setup.bash
```

terminal B 
```
eval "$(/home/vale/mambaforge/bin/conda shell.bash hook)" 
conda activate pan_seg_3d
cd  to/ros/workspace/cotaining/packages
source ./devel/setup.bash
```


In terminal A start the the camera feed

```
roslaunch realsense2_camera rs_camera.launch depth_width:=424 depth_height:=240 depth_fps:=30 color_width:=424 color_height:=240 color_fps:=30 align_depth:=true
```

to test if everything is right in terminal B do:

```
rqt_image_view
```


in terminal B start the main code 
if the terminal B is located on a different machine w.r.t. the main code here you need to setup two environmental variables

```
# this with the IP of the master location
export ROS_MASTER_URI=http://192.168.1.27:11311/  
# this with the IP of the machine where terminal B is hosted
export ROS_IP=192.168.1.27
```

```
rosrun rosrun 3dPan main.py
```

# troubleshoooting
if you get this error:
from this file mambaforge/envs/pan_seg_3d/lib/python3.9/site-packages/arviz/stats/density_utils.py
cannot import name 'gaussian' from 'scipy.signal' (/home/robohikeuser/mambaforge/envs/pan_seg_3d/lib/python3.9/site-packages/scipy/signal/__init__.py)

it is due to an incosistencies with the scipy.signal package where the gaussian has been replaced with window.gaussian
line to add 
from scipy import signal
line to changes 
from scipy.signal import convolve, convolve2d#, gaussian  # pylint: disable=no-name-in-module
#kernel = gaussian(kernel_n, bw)
kernel = signal.windows.gaussian(kernel_n, bw)



      






