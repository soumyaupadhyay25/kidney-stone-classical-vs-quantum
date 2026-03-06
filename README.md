# Quantum vs Classical Kidney Stone Detection using Ultrasound

## Overview

This project compares **Classical Machine Learning** and **Quantum Machine Learning** approaches for detecting kidney stones from ultrasound images.

The goal is to evaluate whether **quantum computing techniques can enhance medical image classification** when compared to traditional machine learning models.

Three systems are evaluated:

1. Classical Model
2. Quantum Simulator
3. Real IBM Quantum Hardware

The results are compared using accuracy, precision, F1 score, and inference time.

---

## Key Idea

Medical imaging tasks are traditionally solved using classical machine learning. This project investigates whether **quantum-enhanced models can perform competitive classification** when applied to ultrasound kidney images.

The pipeline builds both classical and quantum models and compares them directly.

---

## Project Pipeline

Ultrasound Image
↓
Feature Extraction
↓
Dimensionality Reduction
↓

Two parallel pipelines:

### Classical Pipeline

Features → Classical Classifier → Prediction

### Quantum Pipeline

Features → Quantum Encoding → Variational Quantum Circuit → Prediction

Results from both pipelines are compared.

---

## Models Compared

### 1 Classical Model

A classical machine learning classifier trained on extracted ultrasound image features.

Performance:

Accuracy: **96.24%**
Precision: **93.39%**
F1 Score: **96.59%**

---

### 2 Quantum Simulator

A quantum machine learning model simulated using a quantum simulator.

Performance:

Accuracy: **74.15%**
Precision: **76.92%**
F1 Score: **75.08%**

---

### 3 IBM Quantum Hardware

The same quantum model executed on **real IBM Quantum hardware**.

Performance:

Accuracy: **68.06%**
Precision: **70.73%**
F1 Score: **69.34%**

---

## Conclusion


Classical models currently outperform quantum models due to:

* limited qubits
* quantum noise
* hardware constraints

However, the experiment demonstrates how **quantum machine learning can be applied to medical image classification**.

---

## Project Structure

```
Kidney-Stone-Comparison
│
├── classical_model
├── quantum_model
├── dataset
│
├── app.py
├── compare_models.py
├── evaluate_all_models.py
├── calibrate_quantum_predictions.py
│
├── generate_classical_predictions.py
├── generate_quantum_sim_predictions.py
├── generate_quantum_ibm_predictions.py
│
├── classical_predictions.npy
├── quantum_sim_predictions.npy
├── quantum_ibm_predictions.npy
│
├── time_comparison_graph.py
├── time_comparison.png
```

---

## Technologies Used

Python
NumPy
Scikit-learn
Qiskit
IBM Quantum Runtime
Matplotlib
Streamlit

---

## Running the Project

Clone the repository:

```
git clone https://github.com/soumyaupadhyay25/Quantum-Kidney-Stone-Detection.git
cd Quantum-Kidney-Stone-Detection
```

Run model evaluation:

```
python evaluate_all_models.py
```

Generate comparison graphs:

```
python time_comparison_graph.py
```

---

## Visualization

The project generates graphs comparing:

* Accuracy
* Precision
* F1 Score
* Inference Time

These results help analyze the difference between **classical and quantum computing approaches**.

---

## Future Work

Improve quantum circuit design
Use larger datasets
Reduce quantum noise impact
Deploy full system as a **web application**

---
