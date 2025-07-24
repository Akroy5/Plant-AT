# 🌿 Plant‑AT: Hybrid Attention Model for Diagnosing Plant Disease

[![Python Version](https://img.shields.io/badge/python-3.8%2B-green.svg)]()

**Plant-AT** is a lightweight yet powerful hybrid attention-based model for **plant disease classification**, designed for **real-time edge deployment** on devices like **Jetson AGX Orin**. The model combines local and global attention (NAT + SSAT), Inverted Residual Blocks, and a robust fusion strategy — achieving **99.01% disease accuracy** and **99.93% plant accuracy** with only **8.8 M parameters**.

---

## 🧠 Key Features

- ✅ **3‑layer IRB backbone** for efficient feature extraction  
- ✅ **LG‑Attention Transformer** combining SSAT (global) + NAT (local) attention  
- ✅ **Multi-scale fusion block** for merging complementary features  
- ✅ **Hybrid loss** (Cross-Entropy + Focal) to address class imbalance  
- ✅ Designed for **edge deployment** (e.g., Jetson AGX Orin)

---

## 📁 Repository Structure

```bash
Plant‑AT/
├── CODE/                       # Source code
│   ├── PlantAT3L.py            # Model & training script
│   ├── requirements.txt        # Python dependencies
│
├── IMAGES/                     # Diagrams and visualizations
│   └── model_architecture.png
│
├── RESULTS/                    # Training outputs (ignored by Git)
│   ├── checkpoints/            # Model weights
│   └── figures/                # Plots and confusion matrices
│
├── DATA/                       # Dataset folder (ignored by Git)
│
├── README.md
└── .gitignore                  # Ignores DATA/ and RESULTS/

---
## **Hyperparameters**
| Hyperparameter          | Value                   |
| ----------------------- | ----------------------- |
| Image Size              | 224 × 224               |
| Batch Size              | 64                      |
| Epochs                  | 100                     |
| Early Stopping Patience | 15                      |
| Learning Rate           | 1e‑4 (Cosine Annealing) |
| Weight Decay            | 1e‑4                    |
| Optimizer               | AdamW                   |
| Gradient Clipping       | 1.0                     |
| Loss Function           | Cross‑Entropy + Focal   |
| Loss Weighting          | Plant + 3× Disease      |
---
## 📊 **Results on PlantVillage Dataset**
Disease Classification

| Model        | Params  | Acc    | Precision | Recall | F1     | AUC    | MCC    |
| ------------ | ------- | ------ | --------- | ------ | ------ | ------ | ------ |
| **Plant‑AT** | 8.8 M   | 99.01% | 99.06%    | 99.01% | 99.01% | 0.9998 | 0.9894 |
| NAT‑Tiny     | 42.4 M  | 98.94% | 98.97%    | 98.94% | 98.94% | 0.9999 | 0.9886 |
| ResNet‑50    | 23.5 M  | 98.94% | 98.96%    | 98.94% | 98.94% | 0.9994 | 0.9886 |
| VGG‑19       | 139.6 M | 97.03% | 97.12%    | 97.03% | 97.01% | 0.9975 | 0.9682 |
| MobileNetV2  | 3.88 M  | 95.55% | 95.77%    | 95.55% | 95.48% | 0.9961 | 0.9524 |

Plant Type Classification

| Model        | Params  | Acc    | AUC    |
| ------------ | ------- | ------ | ------ |
| **Plant‑AT** | 8.8 M   | 99.93% | 1.0000 |
| NAT‑Tiny     | 42.4 M  | 99.97% | 1.0000 |
| ResNet‑50    | 23.5 M  | 99.93% | 1.0000 |
| VGG‑19       | 139.6 M | 99.79% | 0.9999 |
| MobileNetV2  | 3.88 M  | 99.72% | 0.9932 |
---
## **Quick Start**
git clone https://github.com/Akroy5/Plant-AT.git
cd Plant-AT
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
pip install -r CODE/requirements.txt
python CODE/PlantAT3L.py --data-dir DATA/ --batch-size 64 --epochs 100 --lr 1e-4
---
## **DATA**
DATA/
├── train/
│   ├── Tomato-Bacterial_spot/
│   ├── Grape-Black_rot/
│   └── ...
└── val/
    ├── Tomato-Bacterial_spot/
    └── Grape-Black_rot/
---
## **📜 License & Publication**
The Plant‑AT model and results are based on this Springer Nature publication:

Akash Nagappagol et al., “Plant‑AT: Customized Hybrid Attention Model for Diagnosing Plant Disease”, Springer, 2025
🔗 https://link.springer.com/book/10.1007/978-3-031-93691-3

This work is subject to Springer’s copyright policies — you may use the code for academic and non-commercial purposes only, and you must cite the paper when using it.
---
## **🤝 Citation**
If Plant‑AT is used in your research or projects, please cite:

Nagappagol, A., et al. (2025). Plant‑AT: Customized Hybrid Attention Model for Diagnosing Plant Disease. Springer Nature. 

f - https://link.springer.com/book/10.1007/978-3-031-93691-3

