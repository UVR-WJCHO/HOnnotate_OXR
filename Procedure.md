1. set recorded samples [rgb_orig, depth_orig](1280*780) on dataset folder, and set camera parameters in calibaration folder.

'''
            - dataset
                - 221215_sample
                    - rgb_orig
                        - mas
                        - sub1
                        - sub2
                        - sub3
                    - depth_orig
                        - ...
                    - rgb(will be instantiate)
                    - depth(will be instantiate)
                    - calibration
                        - cam_0_intrinsics.txt
                        - cam_0_depth_scale.txt
                        - ...

'''

2. Preprocess dataset

'''
        cd modules
        python preprocess_db.py --db '221215_sample' --seq 'bowl_18_00' --cam 'mas'
'''

3. Hand+Object segmentation

'''
        cd ../HOnnotate_refine
        python inference_seg.py --db '221215_sample' --seq 'bowl_18_00' --cam 'mas'
'''
