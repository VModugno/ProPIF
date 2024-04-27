# Install ultralytics (yolo and sam)

create a new mamba evn
activate mamba env
mamba install pip
pip install ultralytics


# Install semantic-sam
create new mamba env 
activate mamba env
mamba install pip
pip install torch torchvision torchaudio

search for the right nvidia toolkit version matching you nvidia cuda driver version(to check nvidia toolkit you can run nvcc --version if you want to check nvidia cuda driver you do nvidia-smi) in this webpage: https://anaconda.org/nvidia/cuda-toolkit in my case was:

mammba install nvidia/label/cuda-12.1.0::cuda-toolkit
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
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





