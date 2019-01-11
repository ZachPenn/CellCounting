# CellCounting
This repository contains iPython scripts for counting cells in up to two channels, as well as cell overlap across channels.  User-drawn regions of interest can also be specified.

## Included Files
* **CellCounter.ipynb** is used to batch process a set of images.
* **CellCounter_Optimization.ipynb** allows the user to choose thresholding parameters for cell counting that closely allign with a sample of user's own images that have been counted by hand.

## Package requirements
The iPython scripts included in this repository require the following packages to be installed in your Conda environment:
* python (3.6.5)
* jupyter
* imread
* mahotas(1.4.4)
* numpy(1.14.3)
* pandas(0.23.0)
* matplotlib(2.2.2) 

The following commands can be executed in your terminal to create the environment: 
* conda config --add channels conda-forge
* conda create -n EnvironmentName python=3.6.5 mahotas=1.4.4 pandas=0.23.0 matplotlib=2.2.2 jupyter imread

To subsequently activate this environment and open Jupyter Notebook enter the following commands in the terminal:
* source activate EnvironmentName
* jupyter notebook

## Image requirements
Single channel, 8-bit, .tif images are required.   

## Image J
There are certain tasks relevant to the execution of these scripts (e.g. estimating cell diameter, drawing regions of interest) for which Image J is recommended.  Instructions regarding how to do these tasks in ImageJ are provided in scripts.

## Running Code
After downloading the ipynb files onto your local computer, from the terminal activate the necessary Conda environment and open Jupyter Notebook, then navigate to the files on your computer. The individual scripts contain more detailed instructions.
