pip install --upgrade pip
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# INSTALLING TORCHSORT
export TORCHSORT=0.1.9
export TORCH=pt20
export CUDA=cpu
export PYTHON=cp310

pip install https://github.com/teddykoker/torchsort/releases/download/v${TORCHSORT}/torchsort-${TORCHSORT}+${TORCH}${CUDA}-${PYTHON}-${PYTHON}-linux_x86_64.whl
