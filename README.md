# PROAKTIV: PROtein Analytics for Kinase Therapeutic Inhibitor Variants

PROAKTIV is a comprehensive machine learning pipeline designed to predict the efficacy of Tyrosine Kinase Inhibitors (TKIs)- against specific cancer-associated protein mutations (EGFR, ALK and BRAF). By leveraging state-of-the-art protein language models (ESM2) and deep learning (CNN_RNN, CNN, fingerprints,transformers) models, this project aims to provide a computational decision-support tool for personalized cancer therapy.

## Scientific Background

The treatment of cancer, a leading cause of premature death and a significant economic burden globally, is profoundly complicated by tumor heterogeneity and the evolution of therapeutic resistance. Personalized medicine seeks to address this by tailoring treatments to the molecular drivers of an individual's cancer. This challenge is particularly evident in Non-Small Cell Lung Cancer (NSCLC), the most common and deadliest form of lung cancer, where mutations in key oncogenic kinases—notably the Epidermal Growth Factor Receptor (EGFR), Anaplastic Lymphoma Kinase (ALK), and B-Raf (BRAF)—are critical determinants of patient response to targeted therapies.

Protein mutations, such as single nucleotide polymorphisms (SNPs), insertions, and deletions, can fundamentally alter drug-target interactions, conferring either sensitivity or resistance to specific inhibitors. Predicting the functional impact of these genetic alterations is therefore essential for effective treatment selection. While computational methods have been increasingly applied to this problem, traditional structure-based approaches are often limited by long computational times and the scarcity of high-quality experimental structures for the vast number of possible mutant proteins. This creates a critical need for scalable, sequence-based predictive models that can leverage the wealth of available data to guide clinical decision-making. This project aims to address this gap by developing a comprehensive, automated pipeline for predicting the bioactivity (pIC50) of ligands against wild-type and mutated forms of EGFR, BRAF, and ALK using deep learning. For more details, see [`docs/scientific_background.md`](https://gemini.google.com/app/docs/scientific_background.md "null").

## Repository Structure

This repository is organized to ensure modularity and reproducibility.

```
PROAKTIV/
│
├── .github/      # CI/CD workflows (e.g., automated testing)
├── configs/      # Configuration files for models and data sources
├── data/         # Raw, processed, and external datasets (managed by DVC)
├── docs/         # Detailed documentation and project-related images
├── models/       # Trained model checkpoints (managed by DVC)
├── notebooks/    # Jupyter notebooks for EDA and prototyping
├── src/          # All Python source code for the project
│   ├── data_processing/
│   ├── modeling/
│   ├── training/
│   ├── inference/
│   └── utils/
├── tests/        # Unit and integration tests
│
├── .gitignore    # Files to be ignored by Git
├── Dockerfile    # Docker container definition
├── environment.yml # Conda environment specification
├── README.md     # This file
└── requirements.txt # Pip package requirements

```

## Installation

To set up the project environment, you can use either Conda or a virtual environment with pip.

**1. Clone the repository:**

```
git clone https://github.com/HaroldMate1/PROAKTIV.git
cd PROAKTIV

```

**2. Set up the Conda environment:**

```
conda env create -f environment.yml
conda activate proaktiv

```

**3. (Alternative) Set up with pip:**

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

**4. Pull data and models (if using DVC):**

```
dvc pull

```

## Usage

The core functionalities of the project are accessible via scripts in the `src/` subdirectories.

1. Data Processing:

To run the full data acquisition and processing pipeline:

```
python src/data_processing/main.py --config configs/data_sources.yml

```

2. Model Training:

To train the ESM-2 model with the parameters defined in its config file:

```
python src/training/train_esm.py --config configs/model_configs/esm2_3b_v1.yml

```

3. Inference and Mutation Screening:

To run the in-silico mutation screen using a trained model:

```
python src/inference/predict.py \
    --model_path models/esm2_3b/best_model.pt \
    --mutation_test \
    --output_csv data/results/mutation_screen_results.csv

```

## Contribution Guidelines

Contributions are welcome. Please follow the `fork-and-pull` workflow.

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Before submitting a PR, please ensure your code is formatted with `black` and passes all tests by running `pytest`.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://gemini.google.com/app/LICENSE.md "null") file for details.

## Citation

If you use PROAKTIV in your research, please cite it as follows:

```
Harold Mateo Mojica Urrego/ Chemical Pharmaceutical Biology-University of Groningen (2025). PROAKTIV: PROtein Analytics for Kinase Therapeutic Inhibitor Variants. GitHub. https://github.com/HaroldMate1/PROAKTIV

```
