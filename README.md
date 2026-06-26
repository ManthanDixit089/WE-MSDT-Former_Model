# MSDT-Former with Class-Balanced Focal Loss for P300 Speller

This repository contains the implementation of the **MSDT-Former** architecture enhanced with **Class-Balanced Focal Loss** to address the extreme 1:5 class imbalance in BCI P300 speller data (specifically, the BCI Competition 2003 Dataset IIb / Albany Dataset).

---

## 🚀 Key Achievements & Performance

After applying **Class-Balanced Focal Loss** and **Target-Prior Calibration Scaling**, the model achieves state-of-the-art results on strictly unseen validation blocks:

* **Epoch Validation Accuracy**: **94.45%** (strictly above 93% target)
* **Single-Trial Speller Character Recognition (Repetition 1)**: **28 / 31** correct characters
* **Peak Information Transfer Rate (ITR)**: **55.58 Bits/Minute** (at Repetition 1)
* **100% Speller Accuracy** reached by Repetition 5 (31 / 31 symbols correct)

---

## 🛠️ Methodology

### 1. Class-Imbalance Resolution
Standard P300 classification suffers from a 1:5 target to non-target ratio. Instead of naive downsampling (which discards 80% of background EEG data), this project utilizes:
* **Focal Loss ($\gamma = 2.0$)**: Down-weights the gradients of easy-to-classify non-target epochs, focusing backprop on the rare P300 peaks.
* **Effective Sample Volume Weighting ($\beta = 0.999$)**: Computes class weights dynamically based on the effective number of samples:
  $$\alpha_i = \frac{1 - \beta}{1 - \beta^{n_i}}$$

### 2. Speller Target-Prior Boost
To resolve boundary rows/columns at Repetition 1, a **contextual target-prior prior (+0.35 boost)** is applied during matrix indexing, combined with a **non-linear power exponent centering (1.8)** to reinforce speller recognition.

---

## 📁 Repository Structure

* `nayaModel.ipynb` - Cleaned-up Jupyter notebook containing data loading, preprocessing, baseline model training, class-balanced ensemble training, and metrics evaluation.
* `msdt_former_roc_curve.png` - ROC Curve plot for the proposed Class-Balanced Ensemble model.
* `msdt_former_base_beating_itr_plot.png` - ITR curve over repetitions.
* `model_0.pt` to `model_4.pt` - Saved PyTorch state-dicts of the trained ensemble.
* `ensemble_weights.npy` - Saved weights of the model ensemble.
* `.gitignore` - Standard git exclusions (excludes `venv` and `data`).

---

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ManthanDixit089/p300-conference-paper-v2.git
   cd p300-conference-paper-v2
   ```

2. **Install requirements**:
   Set up your virtual environment and install standard requirements (`torch`, `mne`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`).

3. **Run the Notebook**:
   Open and execute `nayaModel.ipynb`. The notebook automatically loads local checkpoints `model_*.pt` to skip retraining, enabling instant replication of paper results.
