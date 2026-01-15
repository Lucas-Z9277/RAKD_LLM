
# RAKD-LLM: Role-Aware and Knowledge-Decoupled LLM for Cross-City Next POI Recommendation

This repository contains the official implementation of the paper **"Role-Aware and Knowledge-Decoupled LLM for Cross-City Next POI Recommendation"**.

It integrates two key components:

1. **RAKD Framework**: A pipeline for Inter-city User Role Analysis, Intra-city Preference Inference, and Role-Distillation Retrieval Augmentation.
2. **Gain-driven fine-tuning**: An Information Gain-driven fine-tuning objective to prioritize decisive tokens (e.g., POI names) during training.

## 🌟 Architecture Overview

The framework consists of four main phases:

1. **Role Analysis**: Inferring user social roles from schema trajectories.
2. **Preference Inference**: Sliding continual learning for daily batch preferences.
3. **Role Distillation**: Retrieving similar user profiles to augment the current user's context.
4. **Gain-driven fine-tuning**: Fine-tuning Llama-2 with a token-decisiveness weighted loss.

## 🛠️ Requirements
We have provided a `requirements.txt` file for reference in setting up the environment.

## 📂 Project Structure

```text
RAKD-LLM/
├── data/
│   └── ref/nyc/              # Dataset directory
├── model/
│   ├── model_interface.py    # Lightning Module (Modified with IGD Loss)
│   ├── trie.py               # Trie structure for IG calculation
│   └── ...
├── role_analysis.py          # Sec 5.1: Generate User Roles
├── preference_inference.py   # Sec 5.2: Infer User Preferences
├── role_distillation.py      # Sec 5.3: Retrieval Augmentation
├── build_rakd_dataset.py     # Sec 5.4: Construct SFT Data
├── generate_freq.py          # Helper: Generate Item Frequencies for IGD
├── main.py                   # Main Trainer Entry Point
├── train_nyc.sh              # Training Script
├── requirements.txt              # Setting up the environment
└── README.md

```

## 🚀 Pipeline & Usage

To reproduce the results, please follow the data processing pipeline sequentially.

### Step 1: Data Preparation
1.The Foursquare dataset: https://sites.google.com/site/yangdingqi/home/foursquare-dataset

2.The Gowalla dataset: https://drive.google.com/drive/folders/1R87cldpUEMFfRODzloFRiNZohzmOdEmA

Place your raw check-in data (e.g., `nyc.txt`) in the root directory or update the paths in the scripts.

### Step 2: Role & Preference Analysis (LLM-based)

> **Note**: The scripts `role_analysis.py` and `preference_inference.py` contain a `mock_llm_inference` function. You **must** replace this with your actual API call (e.g., API or local Llama-2 inference) to get real analysis results.

**2.1 Inter-city User Role Analysis**
Generates `user_roles_output.json`.

```bash
python role_analysis.py

```

**2.2 Intra-city Preference Inference**
Generates `intra_city_preferences.json`.

```bash
python preference_inference.py

```

### Step 3: Role-Distillation Retrieval

Retrieves similar user candidates to augment the input prompt. Generates `user_distillation_candidates.json` (internal or merged in next step).

```bash
python role_distillation.py

```

### Step 4: Construct RAKD Training Data

Merges the outputs from Steps 2 & 3 with the raw check-in history to create the final Supervised Fine-Tuning (SFT) dataset (e.g., `data/ref/nyc/rakd_train.json`).

```bash
python build_rakd_dataset.py

```

### Step 5: IGD-Tuning Preparation

Calculate the frequency of POI tokens to build the Trie for Information Gain calculation. This generates `item_freq.json`.

```bash
python generate_freq.py

```

## ⚡ Training (IGD-Tuning)

We use **Llama-2-7b** as the backbone. The `main.py` has been modified to support IGD-Tuning via the `--use_igd` flag.

Run the training script:

```bash
bash train_nyc.sh

```

**Key Arguments in `train_nyc.sh`:**

* `--use_igd`: Activates the Information Gain-based Loss.
* `--beta 0.1`: Controls the weight of zero-IG tokens (non-decisive tokens). Recommended value is `0.1`.
* `--item_freq_path ./item_freq.json`: Path to the frequency file generated in Step 5.
* `--dataset`: Ensures the data loader reads the correct RAKD json file.

## 📊 Evaluation

(Assuming `test_nyc.sh` is configured similarly to training but with `--mode test`)

```bash
bash test_nyc.sh

```


## 🙏 Acknowledgements
We thank everyone who has helped our work. Shanghai, China.
