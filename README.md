# MIE1517_nlp_proteins
Research project for MIE1517: Introduction to Deep Learning

## Creating and activating Conda envirnment

To create the conda environment, navigate to the root folder of the repository and run the following code:
```console
conda env create -f ./environments/environment.yml
```
Environment should be created without any conflicts (tested on Linuc and Mac OS).

For the GPU support you need to create GPU-enabled evinment, but running:
```console
conda env create -f ./environments/environmentGPU.yml
```
Note, that GPU environment of the given configuration is only available for Linux OS.

The above will create conda virtual environment, named `PyPorteins` (`PyProteinsGPU` with GPU support). The environment can then be activated by running:
```console
conda activate PyProteins
```
or
```console
conda activate PyProteinsGPU
```
for GPU support.

