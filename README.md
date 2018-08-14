# ccfepyutils 
[![pipeline status](https://git.ccfe.ac.uk/tfarley/ccfepyutils/badges/master/pipeline.svg)](https://git.ccfe.ac.uk/tfarley/ccfepyutils/commits/python3)
[![coverage report](https://git.ccfe.ac.uk/tfarley/ccfepyutils/badges/master/coverage.svg)](https://git.ccfe.ac.uk/tfarley/ccfepyutils/commits/python3)
Central repository for general purpose python classes and functions used at CCFE.

This repository will be updated soon.

To contribute please contact Tom Farley (Tom.Farley@ukaea.uk) and consider
attending [Coding Discussion Group](https://git.ccfe.ac.uk/tfarley/Coding_Discussion_Group) meetings.

### Files contained in repository
**_MODULE_**  | **_DESCRIPTION_**  
---|---
**data**		|	*Collections of general purpose functions for data analysis*


## DEPENDENCIES
### Standard
These packages are all available through pip/conda. 
- Python 3.6+
- numpy
- scipy
- pandas
- xarray
- matplotlib
- cv2
- natsort  (optional, but recommended - may be buggy without)
- PyQt5  (optional, needed for GUI)
- plotly  (optional, needed for some plots)

### CCFE libraries 
These packages are availble internally from [https://git.ccfe.ac.uk](https://git.ccfe.ac.uk).
- [pyEquilibrium](https://git.ccfe.ac.uk/SOL_Transport/pyEquilibrium)
- [cyFieldlineTracer](https://git.ccfe.ac.uk/SOL_Transport/cyFieldlineTracer) / 
  [pyFieldlineTracer](https://git.ccfe.ac.uk/SOL_Transport/pyFieldlineTracer) 
- [pyIpx](https://git.ccfe.ac.uk/SOL_Transport/pyIpx)
- [inference_tools](https://git.ccfe.ac.uk/bayesian_analysis/inference_tools)  (optional, needed for plotting error envelopes on fits)

## SETUP 

Make sure you have your ssh key setup on ccfe gitlab ([guide](https://git.ccfe.ac.uk/help/ssh/README#generating-a-new-ssh-key-pair)). 
Then to recursively download the repository and its CCFE dependencies:

git clone --recursive -j8 git@git.ccfe.ac.uk:tfarley/elzar.git

#### Developer install
To install as a developer (so that the importable module tracks your local changes), from the top level elzar folder containing setup.py run:

pip install -e .

#### User install
To install as a user (fixed version), from the top level elzar folder containing setup.py run:

pip install .