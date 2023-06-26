# Dataset acquisition system for Hand and Object pose
## Introduction
TBD


## Installation

Follow procedures in [HOnnotate](https://github.com/shreyashampali/HOnnotate?)

[Note]
- (old) Environment

```
    - Ubuntu 18.04, GTX1080 ti, CUDA 9.0
    - python 3.6.13, tensorflow-gpu 1.12, gcc 6, g++ 6, tensorflow_probability 0.5.0, PyOpenGL-accelerate
```

- (new) Environment

```
    - Ubuntu 22.04, RTX3090, CUDA 11.8
    - python 3.9.16, torch 2.0.0
    - mediapipe, chumpy, cython, matplotlib, numpy, opencv-python, pillow, scikit-image, scipy, tqdm
```




## Usage

- Download sample dataset in [link](https://www.dropbox.com/s/kztfopvc7rmdab6/230612_samples.zip?dl=0)
- Download mediapipe 2.5D hand results in [link](https://www.notion.so/20230612-6edc4de099ae492397d2727f53c3ae5a)

- Set the dataset structure as below. 
- Make sure both the image folder and the hand result folder have the same sequence name.


```
            - dataset
                - 230612
                    - 230612_bare
                        - rgb
                        - depth
                    - ...
                - 230612_cam
		    - cam_mas_intrinsics.txt
		    - cam_mas_depth_scale.txt
		    - ...
		- 230612_hand
		    - 230612_bare
		        - handDetection_uvd.json
		        - handDetection_xyz.json
		    - ...
```



- run
```
    main.py
```


## Acknowledgement
We borrowed a part of the open-source code of [HOnnotate](https://github.com/shreyashampali/HOnnotate?). 

