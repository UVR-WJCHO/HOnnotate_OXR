

class Config:
    batch_size = 1
    cuda = True
    side = 'right'
    random_shape = True
    rand_mag = 1.0
    flat_hand_mean = True
    iters = 20
    
    mano_root = 'mano/models'
    root_rot_mode = 'axisang' 
    joint_rot_mode = 'axisang'  #choices=['rotmat', 'axisang']
    
    no_pca = False
    mano_ncomps = 45
    
cfg = Config()
