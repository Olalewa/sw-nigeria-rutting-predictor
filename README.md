# Southwest Nigeria Rutting Deformation Predictor

**MSc Research Project — University of Ibadan, Department of Civil Engineering**
**Author:** Apara Olalewa | Matric: 252070 | Supervisor: Dr. Folake Akintayo

---

## Overview

A machine learning surrogate model that predicts rutting deformation 
in flexible pavements of Southwest Nigeria, trained on 500 PLAXIS 3D 
finite element analysis simulations.

This tool enables practising engineers to obtain mechanistic rutting 
predictions in milliseconds — without requiring access to PLAXIS 3D 
or specialist FEA expertise.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Gradient Boosting Regressor |
| Training data | 500 PLAXIS 3D FEA simulations |
| CV R² | 0.9876 ± 0.0025 |
| RMSE | 0.2273 mm |
| MAPE | 6.37% |
| FEA Software | PLAXIS 3D Advanced 2024 |

---

## Input Variables

| Variable | Range | Source |
|----------|-------|--------|
| Subgrade stiffness (E50Ref) | 5,000–150,000 kPa | Ola (1983); Gidigasu (1976) |
| Cohesion (cRef) | 5–50 kPa | Bello (2011) |
| Friction angle (phi) | 18–42° | Gidigasu (1976) |
| Asphalt thickness | 75–250 mm | Federal Highway Manual Nigeria |
| Axle load | 400–900 kN/m² | Nigerian traffic loading data |
| Soil type | 4 SW Nigerian groups | Regional classification |

### Soil Groups Covered
- **Lagos Blue Clay** — coastal zone, CBR 2–8%
- **Soft Laterite** — transitional zone, CBR 8–20%
- **Medium Laterite** — inland plateau, CBR 20–40%
- **Stiff Laterite** — uplands, CBR 40–80%

---

## Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/sw-nigeria-rutting-predictor
cd sw-nigeria-rutting-predictor
pip install -r requirements.txt
```

### Run the prediction tool
```bash
python predict_rut.py
```

### Retrain the model
```bash
python train_model.py
```

---

## Repository Structure
├── predict_rut.py              # Interactive CLI prediction tool
├── train_model.py              # Model training and validation
├── plaxis_automation.py        # PLAXIS 3D FEA automation script
├── rutting_dataset_stratified.csv  # 500-run FEA dataset
├── rutting_rf_model_v2.pkl     # Saved trained model
├── SW_Nigeria_Rutting_Predictor.jsx  # React web interface
└── requirements.txt            # Python dependencies

---

## Methodology

1. **FEA Model** — PLAXIS 3D Advanced 2024, 4×4×4m domain,
   Hardening Soil constitutive model for laterite subgrade
2. **Sampling** — Stratified Latin Hypercube Sampling,
   125 runs × 4 soil groups = 500 total simulations
3. **Automation** — Python plxscripting API, overnight execution
4. **ML Training** — Gradient Boosting Regressor,
   5-fold cross-validation, log-transformed features
5. **Validation** — CV R²=0.9876, RMSE=0.2273mm, MAPE=6.37%

---

## Key Finding

Subgrade stiffness (E50Ref) accounts for **70.98%** of rutting
prediction variance in Southwest Nigerian flexible pavements —
confirming that subgrade characterisation is the most critical
factor in mechanistic pavement design for the region.

---

## CBR Input Support

Engineers without triaxial test data can input standard CBR values.
The tool converts CBR to E50Ref internally using the Powell (1984)
correlation:
E(MPa) = 17.6 × CBR^0.64

---

## Limitations

- Predicts single load application deformation, not cumulative
  rutting over design life
- Asphalt modelled as linear elastic — temperature effects
  not captured
- Base course thickness fixed at 300mm

---

## References

- Ola, S.A. (1983). Geotechnical properties of Nigerian laterites
- Gidigasu, M.D. (1976). Laterite Soil Engineering. Elsevier
- Bello, A.A. (2011). Shear strength of lateritic soils
- Powell et al. (1984). CBR to elastic modulus correlation
- Baecher & Christian (2003). Reliability and Statistics
  in Geotechnical Engineering
- Akintayo & Ibrahim (2024). UICIVIL 2024 Conference, UI

---

## Citation
Olalewa, A. (2026). Predicting Rutting Deformation in Flexible
Pavements of Southwest Nigeria: A Machine Learning-Based Surrogate
Model Derived from 3D Finite Element Analysis.
MSc Thesis, University of Ibadan.

---

## Contact

**Apara Olalewa**
Department of Civil Engineering, University of Ibadan