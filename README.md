# Dataset acquisition system for Hand and Object pose
## Introduction
TBD


## Installation

- Create a conda environment and install the following main packages

```
    - Ubuntu 22.04, RTX3090, CUDA 11.8
    - python 3.9.16, torch 2.0.0
    - mediapipe, chumpy, cython, matplotlib, numpy, opencv-python, pillow, scikit-image, scipy, tqdm
```



## Setup

- Clone the repository and checkout the develop branch
- Download all resources in the link [Dropbox](https://www.dropbox.com/scl/fo/un34gknh23o8sr559j2d3/h?dl=0&rlkey=6ds7v183pp4htjy8hp1kq6wlh)
- Set the dataset structure as below. 
- Make sure both the image folder and the hand result folder have the same sequence name.


```
    - HOnnotate_refine
        - checkpoints
        - models
        - objCorners
    	- ...
    - dataset
        - 230612
        - 230612_cam
	- 230612_hand
    - modules
    - README.md
    - ...
```



- run
```
    main.py
```


## Acknowledgement
We borrowed a part of the open-source code of [HOnnotate](https://github.com/shreyashampali/HOnnotate?). 

