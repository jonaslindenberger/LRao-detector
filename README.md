# LRao-detector
This repository provides the python implementations of two examples for the CNN + LRao detector.

This code was designed as a straightforward implementation rather than a reusable library. It is intended to provide a basic foundation that others can build upon for their own applications.

Software dependencies:

## Requirements

* The code was developed and tested on Windows Server 2022 (Version 21H2) using **Python 3.13.2**. It requires no non-standard hardware and is expected to run on any standard Windows, macOS, or Linux system supporting the required Python dependencies.
To run the scripts and notebooks, the following packages and versions are required:
* `numpy==2.2.5`
* `scipy==1.15.3`
* `matplotlib==3.10.0`
* `torch==2.7.0` 

**Optional but recommended (depending on your specific execution environment):**
* `ipykernel==6.29.5` (Required to run the `.ipynb` notebooks)
* `scikit-learn==1.8.0` (If your notebooks calculate ROC/AUC metrics)
* `tqdm==4.67.3` (If progress bars are used in your training loops)
* `torchinfo==1.8.0` / `torchsummary==1.5.1` (If you print the CNN architecture summary)

