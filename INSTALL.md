## Installation

### Requirements:
- torch==1.10.1+cu111
- torchvision==0.11.2+cu111
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment

conda create -y --name pysgg python=3.9
conda activate pysgg

# this installs the right pip and dependencies for the fresh python
conda install -y ipython scipy h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides gpustat gitpython ipdb graphviz tensorboardx termcolor scikit-learn

# PyTorch installation 
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html


mkdir third_party
cd third_party
export INSTALL_DIR=$PWD

# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd $INSTALL_DIR
cd ..

python setup.py build develop


unset INSTALL_DIR

#in case numpy version not compatible with torch, downgrade numpy 
pip uninstall numpy
pip install numpy==1.23.1
