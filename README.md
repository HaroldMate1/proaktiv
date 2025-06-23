# PROAKTIV: PROtein Analytics for Kinase Therapeutic Inhibitor Variants

PROAKTIV is a comprehensive machine learning pipeline designed to predict the efficacy of Tyrosine Kinase Inhibitors (TKIs)- against specific cancer-associated protein mutations (Mainly focused in EGFR, ALK and BRAF). By leveraging state-of-the-art protein language models and deep learning techniques, this project aims to provide a computational decision-support tool to guide personalized cancer therapy and overcome acquired drug resistance.

## Scientific Background

Acquired resistance to Tyrosine Kinase Inhibitors is a major clinical challenge in oncology. Tumors can develop secondary mutations in the target kinase domain (e.g., EGFR, ALK, BRAF), reducing drug binding affinity and leading to treatment failure. This project aims to create an *in-silico* model that can prospectively predict how these mutations impact drug efficacy, enabling the rational selection of next-generation inhibitors or alternative therapeutic strategies. For more details, see [`docs/scientific_background.md`](https://gemini.google.com/app/docs/scientific_background.md "null").

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
