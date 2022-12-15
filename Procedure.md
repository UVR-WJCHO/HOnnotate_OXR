1. set recorded samples [rgb, depth](1280*780) on dataset folder, and set camera parameters in calibaration folder.

```
            - dataset
                - 221215_sample
                    - calibration
                        - cam_0_intrinsics.txt
                        - cam_0_depth_scale.txt
                    - bowl_18_00
                        - rgb
                        - depth
                    - bowl_18_01
                        - rgb
                        - depth
                    ...

'''

2. run preprocess script

```
        cd modules
        python preprocess_db.py --db '221215_sample' --seq 'bowl_18_00' --cam 'mas'
```