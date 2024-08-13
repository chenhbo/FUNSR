## FUNSR: Neural Implicit Surface Reconstruction of Freehand 3D Ultrasound Volume with Geometric Constraint

Our code is implemented in Python 3.8, PyTorch 1.12.1 and CUDA 11.6

## Usage


### Install Dependencies 
```bash
conda create -n funsr python=3.8
conda activate funsr
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install tqdm pyhocon==0.3.57 trimesh PyMCubes scipy
pip install matplotlib
```

### Data Preparation

- Put the point cloud data on ./data.

- The point cloud data format is in .ply and .xyz.

### Run

```
python run_normalizedSpace.py --gpu 0 --conf confs/conf.conf --dataname case000072.nii_ds  --dir case000072.nii_ds
 ```
