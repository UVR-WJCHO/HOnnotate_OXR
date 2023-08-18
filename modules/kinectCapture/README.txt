Open Anaconda Prompt
activate AzureKinectCapture
cd C:\Users\UVR\Desktop\OXR_WoojinCho\OXR_sampleRecorder

python AzureKinect.py --dir 220630_test
(혹은 AzureKinect.py 내 --dir default 값 수정해서 사용)

python AzureKinect_4cam.py --dir 220704_4cam
(카메라 4대 연결 시)



scp -r 230802_banana donghwan@143.248.47.162:/hdd1/donghwan/OXR/HOnnotate_OXR/dataset/230802