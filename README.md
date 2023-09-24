# Dataset acquisition system for Hand and Object pose

## Introduction
```bash
$ apt-get install -y git
$ git clone https://github.com/UVR-WJCHO/HOnnotate_OXR 
$ cd HOnnotate_OXR
```

## Data Download
- [벡터바이오 NAS](http://quickconnect.to/vectorbio)에서 직접 다운로드
    - Headless의 경우 wget으로 다운로드가 안되어서 vscode를 이용해서 드래그 앤 드롭으로 옮기거나 scp 명령어를 사용
    - scp -v ${로컬 pc 데이터 경로} ${서버 User Name}@${서버 IP}:${서버 dataset 경로}
- 데이터PC에서 다운로드(데이터 PC IP, password 등 엑셀시트에서 확인)
    - scp -v datapc@${데이터 PC IP}:/mnt/download/${원하는 데이터 파일} ${원하는 경로}

scp 옵션
c : 데이터를 압축하여 전송한다.
p(소) : 시간, 접근시간, 모드를 원본과 같도록 전송한다.
r : 디렉터리를 전송한다.
v : 전송과정을 상세히 출력하여 전송한다.

## Installation(Linux server)
도커를 설치하지 않고 로컬에서 설치하여 진행하는 경우 이하 순서 대로 진행
- Conda 설치
```bash
$ wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
$ bash Anaconda3-2023.07-2-Linux-x86_64.sh
```
Path 지정에서 Enter 누른 후, 밑의 문구까지 Space

```bash
Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]
[no] >>> yes
```
yes 입력 후, terminal 재실행. 

- Conda Env 생성
```bash
$ conda create -n oxr_dataset python=3.9.16
$ conda activate oxr_dataset
$ pip install -r requirements.txt 
```

OR

```bash
$ conda env create --file environment.yaml
$ conda activate oxr_dataset
```

- Pytorch & Pytorch3D 설치
Pytorch(1.13~)+cu11이상을 설치
ex) 서버의 cuda 버전이 11.7 이상일 경우
```bash
$ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html
```

- Open3D 설치
```bash
$ pip install open3d
```

Headless 서버의 경우
```bash
$ apt-get install -y libgl1-mesa-glx libglib2.0-0 libegl1 && \
    apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libosmesa6-dev
$ cd
$ git clone https://github.com/isl-org/Open3D 
$ cd Open3D
$ mkdir build && cd build
$ cmake -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF .. 
$ make -j$(nproc) 
$ make install-pip-package
```

cmake에서 오류가 발생하는 경우
```bash
$ wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6.tar.gz
$ tar -zvxf cmake-3.27.6.tar.gz
$ cd cmake-3.27.6
$ ./bootstrap --prefix=$HOME/cmake-install
$ make 
$ make install
$ vim ~/.bashrc
    # vim 에디터에서 이하 내용 추가
    export PATH=$HOME/cmake-install/bin:$PATH
    export CMAKE_PREFIX_PATH=$HOME/cmake-install:$CMAKE_PREFIX_PAT
$ source ~/.bashrc
$ cmake --version
# cmake version 3.27.6 라고 뜨면 정상
```


- [manopth](https://github.com/hassony2/manopth) 설치
```bash
$ cd
$ git clone https://github.com/hassony2/manopth.git
$ cd manopth
$ pip install .
```

- ChamferDistancePytorch([ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/master)) 설치
```bash
$ cd
$ cd HOnnotate_OXR/utils
$ git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
$ cd ChamferDistancePytorch/chamfer3D
$ python setup.py install
$ cd 
```

## Docker setting
도커를 사용하여 진행할 경우
- Docker 설치
Reference 참조 ([http://deepmi.me/linux/2021/01/08/docker/](http://deepmi.me/linux/2021/01/08/docker/))
```bash
# Docker 설치
$ sudo wget -qO- https://get.docker.com/ | sh

# Docker Group에 사용자 추가 (Docker는 기본적으로 root 권한이 필요)
$ sudo usermod -aG docker <username> # e.g. jhk as <username>
$ sudo reboot
## check
$ docker run hello-world

# (Optional) Docker-compose 설치
$ sudo curl -L https://github.com/docker/compose/releases/download/1.21.2/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
$ sudo docker-compose --version

# Nvidia-docker 설치
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
$ sudo systemctl restart docker
```

- Docker 커맨드
```bash
#################################
# 실행중인 container 확인
$ docker ps 

# 모든 container 확인
$ docker ps -a

# 컨테이너 시작/중지
$ docker start COTNAINER_NAME
$ docker stop CONTAINER_NAME

# 실행 중인 컨테이너 진입
$ docker attach CONTAINTER NAME

# 컨테이너를 종료하지 않기 빠져나오기
# (!)중요: 컨테이너는 종료하지 않는 것이 안전. (종료 시, 내부 작업 완료 요망)
ctrl + p + q
#################################
```

- Pull Docker Image (본인 서버 버전에 맏는 이미지)
```
**Process** 

1. [docker hub](http://hub.docker.com)에 접속, 아이디 생성
2. nvidia/cuda로 접근 (https://hub.docker.com/r/nvidia/cuda/tags/)
3. 원하는 cuda/cuDNN version 검색 # 반드시 devel이 포함된 태그를 선택
4. pull command ("docker pull ~") 복사
5. terminal에서 실행 $ docker pull ~
```

- Run docker w/ port forwarding & volume mounting
**Port forwarding**
→ docker container 내부로 편리하게 접속하기 위한 작업 
→ local pc에서도 vscode/terminal을 통해 container에 접근

**Volume mounting**
→ container에서 server 저장공간에 접근하기 위한 작업  
→ 대용량 dataset을 container에 받을 필요 없이, 여러 container에서 서버로부터 공유

```bash
**Docker run command** 

# 이미지 기반 container 시작
$ docker run -it --name {container_name} --gpus '"device={device_num}"' --shm-size={n}G -p={host_port}:{conatiner_port} -v={host_folder_path:container_folder_path} {image_name:tag} /bin/bash 

# ==================================================================>
# 이름 설정
--name {container name} --gpus # ex) my_container

# gpu 할당
--gpus '"device={device_num}"' # ex) '"device=0,1"'

# cpu (shared-memory) 할당
--shm-size = {n}G # ex) 16G

# Port forwarding (중요)
- host(server)의 {host_port}를 통해 container의 {container_port}로 접속
- -p={host_port}:{container_port}
- ex) -p=1111:22
- *(!) 다른 사용자 도커의 port forwarding과 겹치지 않도록 주의 (!)
- (!) 22: system, 6006: tensorboard, 8888: jupyter notebook (!) # 22-03-21 Update*

#	Volume mounting
- host의 directory를 container의 directory로 연결
- -v={path_to_host_directory:path_to_container_directory}
- ex) -v=/mnt/sda/usr/Dataset:/root/Dataset
# <==================================================================
```

ex)
```bash
$ docker run -it --name oxr_cont --gpus '"device=0"' --shm-size=16G -p=8888:22 -v=/mnt/oxr_workplace:/root/oxr_workplace nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04 /bin/bash
```

- 기본 설정
```bash
$ apt update
$ apt-get update
$ apt-get upgrade
$ apt install sudo
$ apt-get install libssl-dev
$ apt-get install -y openssh-server
$ apt-get install -y vim
$ apt-get install -y wget
$ vi /etc/ssh/sshd_config 
----vim 에디터에서
    PermitRootLogin: {...} -> yes로 변경
$ passwd root # root password 설정
```

## ETC
1. tmux 활용법 (프롬프트 나누기)
```bash
$ apt-get install tmux
$ tmux new # 새로운 세션 생성
##세션 내에서
ctrl+b ${PANE_NUM} # PANE_NUM번 창으로 이동
ctrl+b+c # 세션 추가
ctrl+d # tmux 세션 밖으로 나감
ctrl+x # 현재 창 제거

##세션 밖에서
$ tmux attach # tmux 세선으로 돌아감
$ tmux kill-session # tmux 세션 전체 삭제
```

## Setup

- Clone the repository and checkout the develop branch
- Download all resources in the link [Dropbox](https://www.dropbox.com/scl/fo/un34gknh23o8sr559j2d3/h?dl=0&rlkey=6ds7v183pp4htjy8hp1kq6wlh)
- Set the dataset structure as below. 
- Make sure both the image folder and the hand result folder have the same sequence name.

- 처리 [데이터 다운로드 NAS(벡터바이오)](http://quickconnect.to/vectorbio)에서 데이터 다운로드
- [Segmentation model 드롭박스](https://www.dropbox.com/scl/fo/tkdh2bc5uakppo4znqbrr/h?rlkey=5dfewdej7p16mbc0jowxj5lfr&dl=0)에서 checkpoints에서 다운로드해서 modules/deepLabV3plus/checkpoints/ 에 위치
- [Object mesh model 드롭박스](https://www.dropbox.com/scl/fi/t06828s9zgwbfuijtsufn/obj_scanned_models.zip?rlkey=spcx7q7lhitgc1x21wnhopyri&dl=0)에서 obj_scanned_models 다운로드해서 dataset/obj_scanned_models/ 에 위치
- [Calibration 드롭박스](https://www.dropbox.com/scl/fi/oyecpc3vdd80v89empyzp/230905_cam-230908_cam.zip?rlkey=i46tivdj1p21rpfm11qllgaud&dl=0)에서 calibration 결과 다운 후 압축 해제해서 datset/ 에 위치
- 이하 폴더 구조 확인

Before Pre-process
```
    - dataset
        - ${YYMMDD}
			- ${YYMMDD_S#_obj_#_grasp_#}
				- trials_${NUM}
					- depth
					- rgb
            ...
        - ${YYMMDD}_cam
        - ${YYMMDD}_cam_2 (없을 수 있음)
        - ${YYMMDD}_obj
        - obj_scanned_models
            - 01_cracker_box
            ...
    - modules
        - deepLabV3plus
            - checkpoints
                - ${ID}_best_...os16.pth
                ...
    - utils
    - ...
```

## Preprocess
처리하고자 하는 데이터의 날짜가 ${YYMMDD} 인경우, 
```
python preprocess.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam
```
같은 날짜라도 시퀀스 별로 해당하는 cam 폴더가 다를 수 있음.
데이터 처리 현황 탭의 해당되는 cam 폴더 이름 **반드시** 확인

```
python preprocess.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam
python preprocess.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam_2 (만약, 존재한다면)
```

[!] seq {xxxx} has not enough hand results, try with ~~~  와 같은 오류 발생 시

```
python preprocess.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam --seq ${SEQ_NAME} --mp_value 0.85
```

오류가 발생한 ${SEQ_NAME}으로 재시도. 오류 반복 발생 시 무시. 

After Pre-Process
```
    - dataset
        - ${YYMMDD}
			- ${YYMMDD_S#_obj_#_grasp_#}
				- trials_${NUM}
					- depth
					- depth_crop
					- meta
					- rgb
					- rgb_crop
					- segmentation
					- segmentation_deep
					- visualizeMP
				...
        - ${YYMMDD}_cam
        - ${YYMMDD}_obj
    - modules
    - utils
    - ...
```

## Optimization
위의 파일 구조 확인 후, 메인 프로세스 진행. 
처리하고자 하는 데이터의 날짜가 ${YYMMDD} 인경우, 
```
python optimization.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam --start ${START_NUM} --end ${END_NUM}
```

- \${YYMMDD} 폴더 안의 모든 시퀀스를 읽고, \${YYMMDD}_S00\_obj00\_grasp\_00 폴더 리스트 중 START_NUM 번째 폴더 부터 END_NUM 번째 폴더까지 작업하는 방식
- 별도 프롬프트 열어서 START_NUM, END_NUM을 바꿔서 병렬 진행.
- 예) 1번 프롬프트 
    ```
    python optimization.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam --start 0 --end 4
    ```
    2번 프롬프트
    ```
    python optimization.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam --start 4 --end 8
    ```
- RTX3090의 경우 3개 실행 보다 2개 프롬프트 실행하는 것이 효율적이었음.
- 사용하고자 하는 GPU를 지정해주고 싶은 경우
  ```bash
  CUDA_VISIBLE_DEVICES=${GPU_NUM} python optimization.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam --start 0 --end 4
  ```

- GUI가 없는 headless 서버에서 작업을 하는 경우
```
python optimization.py --db ${YYMMDD} --cam_db ${YYMMDD}_cam --start_seq ${START_NUM} --end_seq ${END_NUM} --headless True
``` 
- 최적화 결과 확인은 for_headless_server/ 폴더에 그때 그때 저장되므로 확인 가능. 

## Data Upload
- 생성된 \${YYMMDD}_result 폴더 내의 **각 sequence 별로** 압축해서 ['유클리드소프트' NAS](http://data.labelon.kr/)에 업로드 (벡터바이오 NAS가 아님!)

ex) \${YYMMDD}_S00\_obj00\_grasp\_00.zip, ${YYMMDD}_S00\_obj00\_grasp\_01.zip, ...

## Acknowledgement
We borrowed a part of the open-source code of [HOnnotate](https://github.com/shreyashampali/HOnnotate?). 
