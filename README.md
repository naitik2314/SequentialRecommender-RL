# Sequential Recommender System with Reinforcement Learning

A research-oriented implementation of a sequential recommendation model leveraging deep reinforcement learning techniques to optimize long-term user engagement and diversity.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Environment Setup](#environment-setup)
  - [Install Dependencies](#install-dependencies)
- [Data Preparation](#data-preparation)
  - [Download Raw Data](#download-raw-data)
  - [Preprocessing](#preprocessing)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Simulation](#simulation)
- [Logging & Visualization](#logging--visualization)
- [Project Structure](#project-structure)
- [Experiments & Reproducibility](#experiments--reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Features

- Sequential recommendation via Deep Q-Network (DQN)
- Support for diversity-promoting reward signals
- Customizable user behavior simulator
- Modular codebase for easy experimentation

## Prerequisites

- Python 3.8 or higher
- Git
- `pip` (or `conda`) package manager

## Installation

### Clone the Repository

```bash
git clone https://github.com/naitik2314/SequentialRecommender-RL.git
cd SequentialRecommender-RL
```

### Environment Setup

Create and activate a virtual environment:

Using `venv` (Linux/macOS):
```bash
python3 -m venv env
source env/bin/activate
```

Using `conda`:
```bash
conda create -n seqrecf-rl python=3.8 -y
conda activate seqrecf-rl
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

### Download Raw Data

```bash
mkdir -p data/raw/ml-25m
wget -P data/raw/ml-25m http://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip data/raw/ml-25m/ml-25m.zip -d data/raw/ml-25m
```

### Preprocessing

Run the preprocessing script to clean and prepare datasets:

```bash
python src/data_preprocess.py --input-dir data/raw/ml-25m --output-dir data/processed
```

Outputs:
- `data/processed/ratings_clean.csv`
- `data/processed/movies_enriched.csv`
- `data/processed/item_features.npy`

## Configuration

All hyperparameters and environment settings are controlled via YAML files under `configs/`. Example:

```yaml
batch_size: 64
gamma: 0.99
learning_rate: 1e-4
epsilon_start: 1.0
epsilon_end: 0.1
epsilon_decay: 1e-5
device: cuda
```

## Usage

### Preprocessing Pipeline
```bash
python src/data_preprocess.py 
```

### Training
```bash
python src/train.py 
```

**Note**: Ensure GPU is enabled in `src/train.py`:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
``` 

### Evaluation
```bash
python src/evaluate.py --model-path models/dqn.pth --config configs/default.yaml
```

### Simulation
Run the user simulator to generate interaction traces:
```bash
python simulation/user_simulator.py --config configs/default.yaml
```

## Logging & Visualization

Training logs are written to `runs/`. Launch TensorBoard:

```bash
tensorboard --logdir runs
```

## Project Structure

```plaintext
SequentialRecommender-RL/
├── data/                    # Raw and processed datasets
│   ├── raw/                 # Downloaded MovieLens datasets
│   └── processed/           # Cleaned and preprocessed data
├── models/                  # Trained model checkpoints and network definitions
├── runs/                    # TensorBoard logs and experiment artifacts
├── simulation/              # User behavior simulator
├── src/                     # Core scripts (preprocessing, training, evaluation)
├── requirements.txt         # Python dependencies
└── README.md                # Project overview and setup instructions
``` 

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit changes (`git commit -m "Add feature XYZ"`)
4. Push to branch (`git push origin feature/XYZ`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this implementation in your research, please cite:

> Naitik et al. (2025). Sequential Recommender System with Reinforcement Learning. GitHub repository.

## Contact

Project maintainers:
- Naitik <naitik@wayne.edu>
- Issues: https://github.com/naitik2314/SequentialRecommender-RL/issues
