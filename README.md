# Sequential Recommender System with Reinforcement Learning

A research-oriented implementation of a sequential recommendation model using reinforcement learning techniques. The project provides environment simulations, model training pipelines, and evaluation utilities.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Environment Setup](#environment-setup)
    - [Using python `venv`](#using-python-venv)
    - [Using `virtualenv`](#using-virtualenv)
    - [Using `conda`](#using-conda)
  - [Installing Dependencies](#installing-dependencies)
  - [Download Data](#download-data)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Prerequisites

- Python 3.8 or higher
- Git (for repository cloning)
- One of the following environment managers:
  - python `venv` (built-in)
  - `virtualenv` (via pip)
  - `conda`

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/SequentialRecommender-RL.git
cd SequentialRecommender-RL
```

### Environment Setup

#### Using python `venv`

```bash
python3 -m venv env
source env/bin/activate    # Linux/macOS
# .\env\Scripts\activate # Windows PowerShell
```

#### Using `virtualenv`

```bash
pip install virtualenv
virtualenv env
source env/bin/activate    # Linux/macOS
# .\env\Scripts\activate # Windows PowerShell
```

#### Using `conda`

```bash
conda create -n seqrecf-rl python=3.8 -y
conda activate seqrecf-rl
```

### Installing Dependencies

Once the environment is active, install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Download Data

```bash
mkdir -p data/raw/ml-25m
wget -P data/raw/ml-25m http://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip data/raw/ml-25m/ml-25m.zip -d data/raw/ml-25m
```

#### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force -Path data\\raw\\ml-25m
Invoke-WebRequest -Uri http://files.grouplens.org/datasets/movielens/ml-25m.zip -OutFile data\\raw\\ml-25m\\ml-25m.zip
Expand-Archive -Path data\\raw\\ml-25m\\ml-25m.zip -DestinationPath data\\raw\\ml-25m
```

---

## Usage

1. Configure your simulation parameters in the provided config files or scripts.
2. Run the training script:
   ```bash
   python train.py --config configs/default.yaml
   ```
3. Evaluate a trained model:
   ```bash
   python evaluate.py --model-path path/to/model.pt
   ```

Refer to the [docs](docs/) folder for detailed guides and examples.

---

## Project Structure

```
SequentialRecommender-RL/
├── configs/         # Configuration files
├── data/            # Data loading and preprocessing
├── models/          # Model definitions
├── scripts/         # Training and evaluation scripts
├── utils/           # Utility functions
├── requirements.txt # Project dependencies
└── README.md        # Project overview and setup instructions
```

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, feature requests, or improvements.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
