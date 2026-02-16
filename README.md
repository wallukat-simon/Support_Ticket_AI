# Support Ticket AI

This project implements a complete NLP pipeline for classifying customer support tickets into predefined categories.  
Multiple modeling approaches are explored and compared, ranging from classical machine learning methods to deep learning and transformer-based models.

The final system integrates the best-performing model into a deployable architecture using a REST API and a web-based user interface.

---

## Project Goals

- Build and evaluate different text classification models:
  - Classical machine learning (TF-IDF + classifiers)
  - Deep learning (LSTM)
  - Transformer-based models (DistilBERT)
- Compare model performance across approaches
- Deploy the best-performing model in a production-like setting using:
  - FastAPI (backend)
  - Streamlit (frontend)
  - Docker (containerization)

---

## Repository Structure
```
Support_Ticket_AI/
├── data/                   # Raw and processed datasets
├── models/                 # Trained models (ML, LSTM, Transformer)
├── src/                    # API and UI source code
├── notebooks/              # Exploratory analysis and model training 
├── assets/                 # Images
├── requirements.txt        # Python dependencies
├── requirements_api.txt    # API-specific dependencies
├── requirements_ui.txt     # UI-specific dependencies
├── api.dockerfile          # Docker configuration for deployment
├── ui.dockerfile           # Docker configuration for UI
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # Project documentation
```

---

## Project Phases

### Phase 1: Data Exploration & Preprocessing
- Exploratory data analysis of support ticket texts
- Cleaning, normalization and tokenization
- Analysis of text length distribution and class imbalance
- Export of a cleaned dataset
-   Associated files:
    - [notebooks/split_data.ipynb](notebooks/split_data.ipynb)
    - [notebooks/exploration.ipynb](notebooks/exploration.ipynb)

### Phase 2: Classical Machine Learning
- TF-IDF vectorization
- Handling of class imbalance (oversampling)
- Model training and hyperparameter tuning (e.g. Random Forest, Logistic Regression)
- Evaluation using classification reports and confusion matrices
- Associated files:
  - [notebooks/machine_learning.ipynb](notebooks/machine_learning.ipynb)

### Phase 3: Deep Learning
- Vocabulary construction and sequence encoding
- Padding and truncation of input sequences
- Training an LSTM-based classifier using PyTorch
- Comparison against the classical ML baseline
- Associated files:
  - [notebooks/deep_learning.ipynb](notebooks/deep_learning.ipynb)

### Phase 4: Transformer-Based Model
- Fine-tuning a pretrained DistilBERT model
- Tokenization using a pretrained tokenizer
- Training and evaluation using Hugging Face Transformers
- Selection of the best-performing model
- Associated files:
  - [notebooks/transformer_classification.ipynb](notebooks/transformer_classification.ipynb)

### Phase 5: Deployment (API, UI & Docker)
- Implementation of a REST API using FastAPI
- Integration of both:
  - transformer-based model, and
  - classical ML model
- Implementation of a Streamlit web interface
- Containerization using Docker and Docker Compose
- Separation of frontend (UI) and backend (API) for scalable deployment
- Associated files:
  - [src/app.py](src/app.py)
  - [src/ui.py](src/ui.py)
  - [api.dockerfile](api.dockerfile)
  - [ui.dockerfile](ui.dockerfile)
  - [docker-compose.yml](docker-compose.yml)

---

### Run the Application (API + UI)
If you are just interested in running the application without going through the setup process, you can use the provided Docker configuration to quickly get both the API and UI up and running.

```bash
docker-compose up --build
```
API will be available at:
```bash
http://localhost:8000/docs
```
UI will be available at:
```bash
http://localhost:8501
```

---

## Setup
This section provides instructions for setting up the environment to run the code in this repository. You can choose to use either `conda` or `pip` for managing your Python environment and dependencies. For reference, the tested Python versions were **Python3.11** and **Python3.12**.

Create and activate a virtual environment using `conda`.
### With conda
```bash
conda create -n <env_name> python=3.11 -y
conda activate <env_name>
```

#### 1. Install the required packages:
```bash
conda install numpy=1.26.4 nltk=3.8.1 pandas=2.3.3 scikit-learn=1.8.0 seaborn=0.13.2 matplotlib=3.10.8  imbalanced-learn=0.14.1 transformers=4.57.1 fastapi=0.128.0 joblib=1.5.3 pydantic=2.12.4 streamlit=1.54.0 requests=2.32.5 uvicorn=0.40.0-y
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

### With pip

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

