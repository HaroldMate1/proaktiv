# PROAKTIV: PROtein Analytics for Kinase Therapeutic Inhibitor Variants

PROAKTIV is a comprehensive machine learning pipeline designed to predict the efficacy of Tyrosine Kinase Inhibitors (TKIs)- against specific cancer-associated protein mutations (EGFR, ALK and BRAF). By leveraging state-of-the-art protein language models (ESM2) and deep learning (CNN_RNN, CNN, fingerprints,transformers) models, this project aims to provide a computational decision-support tool for personalized cancer therapy.

## Scientific Background

The treatment of cancer, a leading cause of premature death and a significant economic burden globally, is profoundly complicated by tumor heterogeneity and the evolution of therapeutic resistance. Personalized medicine seeks to address this by tailoring treatments to the molecular drivers of an individual's cancer. This challenge is particularly evident in Non-Small Cell Lung Cancer (NSCLC), the most common and deadliest form of lung cancer, where mutations in key oncogenic kinases—notably the Epidermal Growth Factor Receptor (EGFR), Anaplastic Lymphoma Kinase (ALK), and B-Raf (BRAF)—are critical determinants of patient response to targeted therapies.

Protein mutations, such as single nucleotide polymorphisms (SNPs), insertions, and deletions, can fundamentally alter drug-target interactions, conferring either sensitivity or resistance to specific inhibitors. Predicting the functional impact of these genetic alterations is therefore essential for effective treatment selection. While computational methods have been increasingly applied to this problem, traditional structure-based approaches are often limited by long computational times and the scarcity of high-quality experimental structures for the vast number of possible mutant proteins. This creates a critical need for scalable, sequence-based predictive models that can leverage the wealth of available data to guide clinical decision-making. This project aims to address this gap by developing a comprehensive, automated pipeline for predicting the bioactivity (pIC50) of ligands against wild-type and mutated forms of EGFR, BRAF, and ALK using deep learning. 

## Repository Structure

This repository is organized to ensure modularity and reproducibility.

```
PROAKTIV/
│   .gitignore
│   AUDIT_FINDINGS.md
│   Dockerfile
│   environment.yml
│   LICENSE
│   README.md
│   requirements.txt
│   REVIEWER_COMMENTS.md
│   REVISION_PLAN.md
│
├───data
│       ALK_IC50_all_assays.xlsx
│       BRAF_IC50_all_assays.xlsx
│       egfr_alk_braf_merged.xlsx
│       EGFR_IC50_all_assays.xlsx
│
├───notebooks
│       plots.ipynb
│
├───src
│   ├───data_processing
│   │       dataset.py
│   │
│   ├───inference
│   │       cnn_rnn_prediction.py
│   │       plm_prediction.py
│   │
│   ├───modeling
│   │       cnn_cnn.py
│   │       cnn_cnn_rnn.py
│   │       cnn_transformer.py
│   │       daylight_cnn.py
│   │       daylight_cnn_rnn.py
│   │       daylight_transformer.py
│   │       morgan_cnn.py
│   │       morgan_cnn_rnn.py
│   │       mpnn_transformer.py
│   │       pubchem_cnn_rnn.py
│   │
│   ├───training
│   │       fingerprints_cnn_rnn.py
│   │       fingerprints_plm.py
│   │
│   └───proaktiv              revision package (see AUDIT_FINDINGS.md)
│       ├───curation
│       │       sequences.py  kinase-domain windowing
│       │       filters.py    eligibility rules, replicate aggregation
│       ├───splits
│       │       strategies.py scaffold / variant / combined / temporal splits
│       │       audit.py      leakage audit
│       └───evaluation
│               metrics.py    RMSE-first metrics with bootstrap CIs
│               uncertainty.py MC-dropout calibration and risk-coverage
│
├───configs
│       kinases.yml           sequence windows per target
│       curation.yml          eligibility rules, assay classes, replicates
│
├───scripts
│       verify_windows.py
│       build_dataset.py
│       make_splits.py
│       run_baselines.py
│
├───results                   generated; split manifests are frozen here
│       curation/
│       splits/
│       baselines/
│
├───tests_unit                actual unit tests (pytest)
│
└───tests                     NOTE: result figures per model, not tests
    ├───cnn_cnn
    ├───cnn_cnn_rnn
    ├───cnn_transformer
    ├───daylight_cnn
    ├───daylight_cnn_rnn
    ├───daylight_transformer
    ├───fingerprint_transformer
    ├───morgan_cnn
    ├───morgan_cnn_rnn
    ├───mpnn_transformer
    ├───pubchem_cnn_rnn
    └───selfies_transformer

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

> **Revision in progress.** The peer-review response is being developed on the
> `revision/reviewer-validation` branch. See [`REVIEWER_COMMENTS.md`](REVIEWER_COMMENTS.md),
> [`REVISION_PLAN.md`](REVISION_PLAN.md) and [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md).
> Results produced before that branch used a random data split and a protein
> encoder that truncated ALK sequences before the mutated residue; they should
> not be relied on. `AUDIT_FINDINGS.md` explains both.

### Curation and validation pipeline

Run in order. Each step writes machine-readable outputs under `results/`.

```bash
# Confirm every mutated residue survives the kinase-domain window
python scripts/verify_windows.py

# Apply eligibility rules, window sequences, aggregate replicates,
# and emit Table 1, the assay-heterogeneity breakdown and the noise floor
python scripts/build_dataset.py

# Freeze split manifests and audit them for leakage
python scripts/make_splits.py

# Reference baselines across every split scheme, RMSE-first
python scripts/run_baselines.py

# Unit tests
python -m pytest tests_unit -q
```

Configuration lives in [`configs/kinases.yml`](configs/kinases.yml) (sequence
windows) and [`configs/curation.yml`](configs/curation.yml) (eligibility rules,
assay classes, replicate policy). Eligibility rules are declared before any
model is fit, and every excluded record is logged with its reason to
`results/curation/exclusions.csv`.

### Model training and inference

The training and inference entry points are the scripts under `src/training/`
and `src/inference/`:

```bash
python src/training/fingerprints_cnn_rnn.py     # Morgan + CNN-RNN benchmark
python src/training/fingerprints_plm.py --help  # ESM2 + fingerprint model
python src/inference/plm_prediction.py --help   # inference and mutation screen
```

`src/training/fingerprints_plm.py` requires a GPU; the ESM2-3B configuration was
trained on HPC. Note that these two scripts still perform their own random split
internally and have not yet been rewired to consume the frozen manifests in
`results/splits/`.

## Contribution Guidelines

Contributions are welcome. Please follow the `fork-and-pull` workflow.

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Before submitting a PR, please ensure your code is formatted with `black` and passes all tests by running `python -m pytest tests_unit -q`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PROAKTIV in your research, please cite it as follows:

```
Harold Mateo Mojica Urrego/ Chemical Pharmaceutical Biology-University of Groningen (2025). PROAKTIV: PROtein Analytics for Kinase Therapeutic Inhibitor Variants. GitHub. https://github.com/HaroldMate1/PROAKTIV

```
