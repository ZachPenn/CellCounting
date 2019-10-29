# CellCounting
This repository contains iPython scripts for counting cells in up to two channels, as well as cell overlap across channels, in 8-bit .tif images.  User-drawn regions of interest can also be specified.

## Included Files
* **CellCounter.ipynb** is used to batch process a set of images.  Results of individual images can also be viewed.
* **CellCounter_Optimization.ipynb** allows the user to choose thresholding parameters for cell counting that closely allign with a sample of user's own images that have been counted by hand.
* **ROIdrawer.ipynb** allows the user to draw regions of interest on an image and save 8bit masks, to be used with CellCounter.ipynb to restrict cell counts in an image to a particular region of interest.
* **CellCounter_Functions.py** contains functions upon which ipynb files depend

![Example](Images/Example.png)

## Package requirements
The iPython scripts included in this repository require the following packages to be installed in your Conda environment:
* python (3.6.5)
* jupyter
* imread
* mahotas(1.4.4)
* numpy(1.14.3)
* pandas(0.23.0)
* matplotlib(2.2.2) 
* holoviews
* scipy
* opencv

The following commands can be executed in your terminal to create the environment: 
* conda config --add channels conda-forge
* conda create -n CellCounter python=3.6.5 mahotas=1.4.4 pandas=0.23.0 matplotlib=2.2.2 jupyter imread holoviews scipy opencv

To subsequently activate this environment and open Jupyter Notebook enter the following commands in the terminal:
* source activate CellCounter
* jupyter notebook

## Image requirements
Single channel, 8-bit, .tif images are required. 
16 and 32 bit image processing will likely soon be added, but I'm lazy for now.  Let me know if you need this.

## Image J
There are certain tasks relevant to the execution of these scripts (e.g. estimating cell diameters, drawing regions of interest) for which Image J is recommended.  Instructions regarding how to do these tasks in ImageJ are provided in the ipynb files..

## Running Code
After downloading the ipynb files onto your local computer, from the terminal activate the necessary Conda environment and open Jupyter Notebook, then navigate to the files on your computer. The individual scripts contain more detailed instructions.
