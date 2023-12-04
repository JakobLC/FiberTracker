This is a repository for fiber growth analysis.

Visual studio code is recommended for development, and running the jupyter notebooks requires the jupyter extension for visual studio code
To create a conda environment for the repository run the following in a terminal with Anaconda installed (google how to install Anaconda if you don't have it):
```
conda create -n fiber-env python=3.10.11
conda activate fiber-env
pip install -r requirements.txt
pip install git+https://github.com/JakobLC/jlc.git
```
installing requirements might throw some errors, but for packages we don't use, so it should be fine.
Afterwards you should be able to run cells in `scripts.ipynb`

