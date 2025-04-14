# ECG Classification

This project implements classification models for distinguishing between healthy and atrial fibrillation (AF) ECG signals using real and synthetic data. The goal is to leverage different classification algorithms to classify ECGs based on extracted features.
This is a use case for the main project here - https://github.com/dieterich-lab/ecg_data_synthesis
The main feature extraction pipeline to get the basic ECG features for real and synthetic data is implemented here - https://github.com/dieterich-lab/ecg_processing

## Installation

1. Clone the repository

```bash
git@github.com:dieterich-lab/ecg_processing.git
cd ecg_processing
```

2. To install the necessary dependencies, you can use:

```bash
pip install -r requirements.txt
```

## Usage

The script supports two main tasks: **real vs. synthetic ECG classification and healthy vs. AF classification**. 
You can run the script from the command line with different parameters to perform these tasks.
Example commands for both the tasks are shown below.

1. Real vs Synthetic Classification:
This command will classify real vs synthetic ECG signals for the healthy label, using peak-based features.
    
```bash
python classify.py --task real_vs_synth --label healthy --use_peaks True
```
**Parameters:**
* --task real_vs_synth: Specifies that the task is to classify real vs synthetic ECG signals.
* --label healthy: Specifies that the classification is for the "healthy" label (you can also use af for atrial fibrillation).
* --use_peaks True: Indicates whether to use peak-based features for classification.

2. Healthy vs AF Classification (with real data):
This command will classify healthy vs AF ECGs using real data.

```bash
python classify.py --task healthy_vs_af --data real --real_csv_path path/to/real_data.csv --real_labels_path path/to/real_labels.npy
```
**Parameters:**
* --task healthy_vs_af: Specifies that the task is to classify healthy vs AF ECGs.
* --data real: Indicates that the data used for classification is real.
* --real_csv_path: The path to the CSV file containing features for real ECG data.
* --real_labels_path: The path to the .npy file containing labels for real ECG data.

3. Healthy vs AF Classification (with synthetic data):

```bash
python classify.py --task healthy_vs_af --data synth --synth_csv_path path/to/synth_data.csv --synth_labels_path path/to/synth_labels.npy
```
**Parameters:**
* --task healthy_vs_af: Specifies that the task is to classify healthy vs AF ECGs.
* --data synth: Indicates that the data used for classification is synthetic.
* --synth_csv_path: The path to the CSV file containing features for synthetic ECG data.
* --synth_labels_path: The path to the .npy file containing labels for synthetic ECG data.
