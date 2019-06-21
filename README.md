# Table of Content
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Workflow](#workflow)

# Introduction
Package to study ATLAS performances electron identification. This package was originally developed for tau leptons (see the origin of the fork for the details). This branch is dedicated to the development of computer vision algorithms.

# Setup 
## Getting started
```bash
git clone https://github.com/mociduki/tauperf.git
cd tauperf
git checkout -b el_image origin/el_image
source setup_lxplus.sh #for lxplus
source setup_lps.sh #for lps local servers
```

<!---
## Getting started on techlab-gpu-nvidiak20-03
```bash
cd /tmp/${USER}
git clone https://github.com/qbuat/tauperf.git
cd tauperf
git checkout -b imaging origin/imaging
source setup_cern_gpu.sh
```
-->

<!---
## Install using a virtual environment

### virtual environment
```bash
virtualenv imaging_ve
source imaging_ve/bin/activate
```
### root setup
you need a working setup of ROOT 6.

### dependencies
note that some of these packages evolve very quickly so the version used can be quite deprecated
```bash
pip install pip --upgrade
pip install theano==0.9.0
pip install keras==2.0.6
pip install pydot_ng==1.0.0
pip install h5py==2.6.0
pip install tables==3.3.0
pip install scikit-learn==0.19.0
pip install scikit-image==0.12.3
pip install matplotlib==1.5.3
pip install root_numpy==4.5.2
pip install rootpy==0.8.3
pip install tabulate==0.7.5
```
### tauperf project: imaging branch
```bash
git clone https://github.com/qbuat/tauperf.git
cd tauperf
git checkout -b imaging origin/imaging
```
# Usage
## Creating your own setup script
1. Copy the [setup](setup_quentin.sh) file
1. Edit the ROOT setup
1. Edit the variables `DATA_AREA` and `VE_PATH` 
-->

## Data (as of Jun. 20th, 2019)
### h5 file extracted from DAOD file by the Copenhagen group
Avaiable only on lxplus
```
/eos/user/l/lehrke/Data/Data/2019-05-02/MC_abseta_0.0_1.3_et_0.0_1000000.0_processes_pid.h5
```

### hdf5 files containing the formated images
On lxplus:
```
/afs/cern.ch/work/user/m/mociduki/el_images
```
On lps:
```
/lcg/storage17/atlas/mociduki/el_images
```
Prepare the symbolic links in your working directory
```
mkdir el_images; cd el_images
ln -s /afs/cern.ch/work/m/mociduki/public/el_images #lxplus
ln -s /lcg/storage17/atlas/mociduki/el_images #lps

ln -s el_images/stat_10k/* .
cd ..
```

<!--
## Processing/training/testing
see the [workflow](doc/workflow.md)
-->


# Workflow
Each time you login, you need to source the setup script you sourced for the first time (see above at setup).

To execute training
```
python fitter_dense_multi.py --one-prong-only --overwrite
```

Note: the first line of the output must be
```
Using Theano backend.
```
If you see that the TensorFlow backend is used instead, you should check your keras config file. Mine looks like below:
```
> cat $HOME/.keras/keras.json
{
    "epsilon": 1e-07, 
    "floatx": "float32", ython fitter_dense_multi.py --one-prong-only --overwrite
    "image_data_format": "channels_last", _multi.py --one-prong-only --overwrite 
    "backend": "theano"
}
```

# Conversion of the h5 file to training-compatible format
```
python transform_el_images.py 0
```
The argument above corresponds to the label number [0,1,2].
To change the input file, look up 'input_filename' directly written in the code.
