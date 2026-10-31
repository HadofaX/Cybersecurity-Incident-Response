# Cybersecurity Incident Response Under Uncertainty

A research-oriented simulation of risk-aware cybersecurity incident response under incomplete information. The project combines Bayesian belief updating, decision theory, Value of Information (VoI), MDP/POMDP-style decision-making, and LLM-generated recommendations.

> **Scope:** This repository contains an academic simulation and benchmark, not a production security product. The scenarios and model parameters are simulated/assumed; results should not be interpreted as validation on real-world SOC incident data.

## Research objectives

- Estimate hidden incident states from uncertain security observations.
- Compare rule-based, probabilistic, decision-theoretic, and LLM-assisted approaches.
- Evaluate the value and cost of additional security checks.
- Study response decisions while accounting for uncertainty, operational cost, and potential harm.

## Experiment design

- 12 security scenarios
- 9 decision models (M0–M8)
- 5 repetitions per scenario/model
- 540 total experimental runs
- Random seed: `42`
- Discount factor (`gamma`): `0.90`
- Maximum permitted steps: `5`

The M4 Oracle MDP is an upper-bound benchmark that assumes access to the true state. It is not directly comparable to deployable models that must infer the state from observations.

## Repository structure

```text
Cybersecurity-Incident-Response/
├── csv/             # Input tables and experiment outputs
├── figures/         # Generated figures and visual summaries
├── *.ipynb          # English and Turkish analysis notebooks
├── experiment_solution.py
├── deney_solution-tr.py
├── experiment_results.xlsx
├── Deney_Tum_Sonuclar-tr.xlsx
├── Cyber_Incident_Response_Report.docx
├── Cybersecurity_Incident_Raporu_Tamamlanmis-tr.docx
└── README.md
```

## Getting started

Use Python 3.10+.

```bash
pip install pandas matplotlib openpyxl jupyter
```

### Run the experiment

From the repository root:

```bash
python experiment_solution.py
```

The script writes generated CSV outputs into `csv/`.

### Run the analysis notebooks

Open either notebook from the repository root using Jupyter:

- `Cybersecurity_Incident_Response_Analysis.ipynb` — English
- `Belirsizlik Altinda Siber Olay Mudahalesi_Analiz_Calistirilmis-tr.ipynb` — Turkish

The notebooks read the prepared tables from the `csv/` directory.

## Key methodological notes

- M4 is an oracle benchmark and assumes knowledge of the true hidden state.
- Model outputs depend on the assumptions, transition/observation probabilities, costs, and scenario definitions encoded in the experiment.
- Results are simulation-based and require evaluation on representative real or carefully constructed semi-synthetic security data before any operational claims can be made.

## Technologies

Python, Pandas, Matplotlib, Jupyter Notebook, Bayesian Networks, Bayesian inference, Expected Utility, Value of Information, MDP, POMDP, and LLM-assisted decision support.

## Author

Computer Engineering Student

## License

No license has been applied yet. All rights reserved unless a license file is added.
