# PAIR-AWAN: Annotation-Wise Attention Network for Hate Speech Detection

This repository implements a **demographic-aware Annotation-Wise Attention Network (AWAN)** for hate speech detection.  
It leverages multiple annotators’ demographic features (age, ethnicity, education, political party) to train per-annotator classifiers and analyze variation in label behavior.

---

## Project Structure
```
PAIR-AWAN/
│
├── src/
│   ├── data_loader.py       # Data reading and preprocessing
│   ├── model_awan.py        # Model architecture (BERT + AWAttention)
│   ├── metrics.py           # Evaluation metrics (Average MD, Error Rate)
│   └── train_hate.py        # Training and evaluation script
│
├── data/splits/             # Place train/dev/test CSVs here
├── results/                 # Model outputs (auto-generated)
├── requirements.txt
└── README.md
```

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ntahaei/PAIR-AWAN.git
   cd PAIR-AWAN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place the CSVs `train_hate.csv`, `dev_hate.csv`, and `test_hate.csv` under `data/splits/`.

---

##  Training

Run the model:
```bash
python src/train_hate.py
```

The script:
- Loads annotated hate-speech data with demographic metadata
- Trains AWAN models across demographic feature pairs
- Evaluates using macro-F1, Precision, Recall, and soft-label metrics (Average MD, Error Rate)

---

## 📎 License
This project is provided for academic and research use under the MIT License.
