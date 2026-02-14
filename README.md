# ALL INTERACTIONS MATTER: MODALITY-PRESERVING DEEP FUSION FOR MULTIMODAL SENTIMENT ANALYSIS


DMPF addresses the challenge of modality suppression in deep fusion by introducing modality-specific learnable tokens and a custom attention masking strategy. This ensures that unique signals from Audio (A), Visual (V), and Audio-Visual (AV) streams are preserved while allowing for controlled cross-modal interactions within a pre-trained Language Model (LM) backbone.

---

## 🏛️ Architecture Overview



The DMPF framework consists of three main components:
1. **Modality-Specific Encoders:** Independent Transformer-based encoders for Audio (A), Visual (V), and joint Audio-Visual (AV) features.
2. **Fused Language Model:** A frozen GPT-2 backbone augmented with **MM Blocks** that inject multimodal context via Gated Cross-Attention (GCA) using separate fusion tokens.
3. **Cross-Modal Self-Attention Task Head:** A late-fusion mechanism that aligns seven heterogeneous representations to prevent modality interference.

---

## 🚀 Performance Highlights

| Dataset | Acc2 ↑ | F1 ↑ | MAE ↓ | Corr ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **MOSEI** | **87.89%** | 87.85 | 0.492 | 0.810 |
| **SIMS** | **83.59%** | 83.87 | 0.346 | 0.735 |

* **Custom Attention Mask:** A novel masking scheme that restricts text tokens from dominating and enforces bimodal interactions through dedicated A and V tokens.
* **Late Fusion:** The Self-Attention Task Head provides a significant performance boost (up to **5% in Acc5**) compared to standard linear concatenation.
* **Token Capacity:** Increasing multimodal capacity from 16 to 34 tokens via modality-specific separation yields consistent gains.

---

## ⚙️ Configuration & Hyperparameters

All hyperparameters—including the number of fusion tokens, MM block insertion depths, and loss weights ($\lambda_m$)—are defined in the YAML files located in the `configs/` directory.

| Parameter | MOSEI | SIMS |
| :--- | :--- | :--- |
| **Backbone** | GPT-2 Large | GPT-2 Base |
| **Fusion Tokens** | $n_a=10, n_v=10, n_{av}=12$ | $n_a=10, n_v=8, n_{av}=16$ |
| **Auxiliary Weights** | $\lambda_a=1.0, \lambda_v=1.0$ | $\lambda_a=0.5, \lambda_v=0.5$ |

---

## 🛠️ Usage

### Installation
```bash
git clone [https://github.com/nickxir12/All-Interactions-Matter.git](https://github.com/nickxir12/All-Interactions-Matter.git)
cd DMPF
pip install -r requirements.txt
```
