# Dataset acquisition system for Hand and Object pose
## Introduction
TBD

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

- Set recorded dataset as following format. For example, if target db is "230104", which includes sequences "bowl_18_00", "bowl_18_01", "apple_28_00", ...
```
	dataset/230104
		bowl_18_00
		bowl_18_01
		apple_28_00
		...
	dataset/230104_calibration
		cam_mas_depth_scale.txt
		cam_mas_intrinsics.txt
		...
	dataset/230104_hand
		hand_result
		mas_intrinsic.json
		...
``` 


- run
```
    modules/preprocess_db.py
    HOnnotate_refine/inference_seg.py
    HOnnotate_refine/optimization/handPoseMultiview.py


    ...
```



## Acknowledgement
We borrowed a part of the open-source code of [HOnnotate](https://github.com/shreyashampali/HOnnotate?). 

