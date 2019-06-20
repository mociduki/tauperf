# Introduction
Package to study ATLAS performances for tau leptons. 
This branch is dedicated to the development of computer vision algorithms.

# Table of Content
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)

# Setup 
## Getting started on lxplus
```bash
git clone https://github.com/mociduki/tauperf.git
cd tauperf
git checkout -b el_image origin/el_image
source setup_lxplus.sh
```

## Getting started on techlab-gpu-nvidiak20-03
```bash
cd /tmp/${USER}
git clone https://github.com/qbuat/tauperf.git
cd tauperf
git checkout -b imaging origin/imaging
source setup_cern_gpu.sh
```

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

## Data (as of Apr. 18th, 2018)
### Flat root ntuples:
```
user.qbuat.mc16_13TeV.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight.recon.ESD.e5468_s3170_r9466.tauid.v2_OUT
```

### root and hdf5 files with selected tau candidates:
```
/eos/atlas/user/q/qbuat/IMAGING/v13/test/output.selected.root
/eos/atlas/user/q/qbuat/IMAGING/v13/test/output.selected.h5
```

### hdf5 files containing the formated images
```
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_1p0n.h5
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_1p1n.h5
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_1p2n.h5
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_3p0n.h5
/eos/atlas/user/q/qbuat/IMAGING/v13/test/images_new_3p1n.h5
```

## Processing/training/testing
see the [workflow](doc/workflow.md)
