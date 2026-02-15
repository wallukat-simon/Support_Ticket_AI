# Setup
This section provides instructions for setting up the environment to run the code in this repository. You can choose to use either `conda` or `pip` for managing your Python environment and dependencies. For reference, the tested Python versions were Python3.11 and Python3.12.

Create and activate a virtual environment using `conda`.
## With conda
```bash
conda create -n <env_name> python=3.11 -y
conda activate <env_name>
```

#### 1. Install the required packages:
```bash
conda install numpy=1.26.4 nltk=3.8.1 numpy-base=1.26.4 pandas=2.3.3 scikit-learn=1.8.0 seaborn=0.13.2 matplotlib=3.10.8  imbalanced-learn=0.14.1 transformers=4.57.1 -y
```

#### 2. Install PyTorch separately based on your system configuration:
Without CUDA support:
```bash
conda install pytorch cpuonly -c pytorch -y
```
With CUDA support:
```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Check the installation:
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

#### 3. Install Juypyter  and IPykernel to use the environment in Jupyter Notebooks:
```bash
conda install jupyter nb_conda_kernels ipykernel -y
```

Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name=pytorch311 --display-name="Conda Env"
```



## With pip

#### 1. Create and activate a virtual environment:
```bash
python -m venv <env_name>
source <env_name>/bin/activate  # On Windows: <env_name>\Scripts\activate
```
Install the required packages:
```bash
pip install -r requirements.txt
```

#### 2. Install PyTorch separately based on your system configuration:
Without CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
With CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Check the installation:
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

#### 3. Install Jupyter and IPykernel to use the environment in Jupyter Notebooks:
```bash
pip install jupyter ipykernel
```

Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name=pytorch311 --display-name="Pip Env"
```

