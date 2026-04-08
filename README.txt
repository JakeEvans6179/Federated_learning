#Install python3.10 environment
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv -y
 
#Create and activate virtual env
python3.10 -m venv .venv
source .venv/bin/activate
 
#Install TensorFlow, Project Libraries, and Matplotlib
pip install tensorflow==2.15.0 pandas numpy scikit-learn pyarrow tqdm matplotlib
 
#Install Flower Federated Learning (lock keras version to 2.15.0)
pip install "flwr[simulation]==1.7.0" "keras<2.16"

#Install NVIDIA CUDA dependencies for TF 2.15 GPU support
pip install nvidia-cublas-cu12==12.2.5.6 nvidia-cuda-cupti-cu12==12.2.142 nvidia-cuda-nvcc-cu12==12.2.140 nvidia-cuda-nvrtc-cu12==12.2.140 nvidia-cuda-runtime-cu12==12.2.140 nvidia-cudnn-cu12==8.9.4.25 nvidia-cufft-cu12==11.0.8.103 nvidia-curand-cu12==10.3.3.141 nvidia-cusolver-cu12==11.5.2.141 nvidia-cusparse-cu12==12.1.2.141 nvidia-nccl-cu12==2.16.5 nvidia-nvjitlink-cu12==12.2.140
 
#Run check to see if it reads GPU
python -c "import tensorflow as tf; print('\nNum GPUs Available:', len(tf.config.list_physical_devices('GPU')))"