FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential g++ git && \
    apt-get install -y wget 

RUN pip install numpy==1.23.1 tqdm tqdm-multiprocess matplotlib

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install mediapipe chumpy cython opencv-python pillow scikit-image scipy

RUN pip install fvcore visdom loguru natsort transforms3d trimesh

RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html

RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libegl1 && \
    apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libosmesa6-dev

RUN cd / && \
    git clone https://github.com/isl-org/Open3D && \
    mkdir /Open3D/build && \
    cd /Open3D/build && \
    cmake -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF .. && \
    make -j$(nproc) && \
    make install-pip-package

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd / && \
    git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git && \
    cd ChamferDistancePytorch/chamfer3D

RUN cd / && \
    git clone https://github.com/hassony2/manopth.git && \
    cd manopth && \
    pip install .

# cd ChamferDistancePytorch/chamfer3D && python3 setup.py install