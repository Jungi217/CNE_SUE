# CNE_SUE

This repository implements [CNE-SUE (Mao et al., 2021)](https://aclanthology.org/2021.findings-emnlp.5/) based on the original NNR code. It supports MIND, Adressa, and EB-NeRD through a shared training and evaluation pipeline.

## Setup

Use Python 3.10 and an NVIDIA GPU. Install the CUDA 11.3 dependencies in a virtual environment.

~~~sh
python -m pip install torch==1.12.1+cu113 torchtext==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
python -m pip install numpy==1.26.4 scikit-learn nltk tqdm
~~~

Install Polars for EB-NeRD conversion with `python -m pip install polars==2.0.0`.

## Datasets

| Dataset | Preparation and splits |
| --- | --- |
| [MIND](https://msnews.github.io/) | Microsoft news impression logs. Follow the [NNR dataset setup](https://github.com/Veason-silverbullet/NNR#dataset-preparation) for MIND-small, MIND-200k, or MIND-large. |
| [Adressa](https://reclab.idi.ntnu.no/dataset/) | Norwegian news click logs. Put the original `contentdata/` and `one_week/` folders under `data/adressa/`. The converter writes `data/mind/` and uses days 1 to 5 for click history and day 6 for training. Day 7 is split into 30% validation and 70% test. |
| [EB-NeRD](https://github.com/jppol-ai/ebnerd-benchmark) | Danish news impression and click logs. Put `articles.parquet`, `train/`, and `validation/` under `data/ebnerd/raw/`. The converter writes TSVs to `data/ebnerd/`, uses the last approximately 10% of official training impressions for validation, and uses the official validation split as the local test set. |

## Run

Run the commands from the repository root. Convert Adressa or EB-NeRD once before training.

~~~sh
python adressaToMIND.py
python main.py --dataset=adressa

python ebnerdToMIND.py
python main.py --dataset=ebnerd

python main.py --dataset=small
~~~

Use `--raw-dir` with `ebnerdToMIND.py` if the original Parquet files are stored elsewhere. Adressa and EB-NeRD results are saved under `runs/cne-sue/`. Local evaluations report AUC, MRR, nDCG@5, and nDCG@10.

Please cite the [CNE-SUE paper](https://aclanthology.org/2021.findings-emnlp.5/) and the dataset used in your experiments.
