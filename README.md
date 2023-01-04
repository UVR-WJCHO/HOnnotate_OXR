# Dataset acquisition system for Hand and Object pose
## Introduction


## Structure

```
            - HOnnotate_refine
                - inference_seg.py
                - ...
            - modules
                - calib.py
                - preprocess_db.py
                - utils
```

need to prepare dataset as below
```
            - dataset
                - 230104
                    - calibration
                        - cam_mas_intrinsics.txt
                        - cam_mas_depth_scale.txt
                        - ...
                    - rgb_orig
                    - depth_orig

                - 230104_hand
                    - hand_result
                        - bowl_18_00
                        - ...
```


## Installation

Follow procedures in [HOnnotate](https://github.com/shreyashampali/HOnnotate?)

[Note]
- Environment

```
    - Ubuntu 18.04, GTX1080 ti, CUDA 9.0
    - python 3.6.13, tensorflow-gpu 1.12, gcc 6, g++ 6, tensorflow_probability 0.5.0, PyOpenGL-accelerate
```

- For dirt, install in development mode
```
    cd dirt
    mkdir build ; cd build
    cmake ../csrc -DCMAKE_CUDA_ARCHITECTURES=61
    make
    cd ..
    pip install -e .
    ...
```






## Usage


```
    modules/preprocess_db.py
    HOnnotate_refine/inference_seg.py
    HOnnotate_refine/optimization/handPoseMultiviewInit.py


    ...
```



## Acknowledgement
We borrowed a part of the open-source code of [HOnnotate](https://github.com/shreyashampali/HOnnotate?). 

