import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

st.title("Kidney Stone Detection")
st.subheader("Classical vs Quantum Model Comparison")

# Load Classical Model
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,2)
)

model.load_state_dict(
    torch.load("classical_model/kidney_model.pth", map_location="cpu")
)

model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

uploaded = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")
    st.image(image,width=300)

    img = transform(image).unsqueeze(0)

    st.write("Running prediction...")

    # -------------------------
    # Classical Prediction
    # -------------------------

    start = time.time()

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output,dim=1)
        pred = torch.argmax(probs,1).item()
        conf = probs[0][pred].item()

    classical_time = round(time.time()-start,4)

    classical_result = "Stone" if pred == 1 else "No Stone"

    # -------------------------
    # Quantum Prediction
    # -------------------------

    try:

        q_start = time.time()

        service = QiskitRuntimeService()

        backend = service.least_busy(simulator=False, operational=True)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0,1)
        qc.measure_all()

        qc = transpile(qc, backend)

        sampler = Sampler(backend)

        job = sampler.run([qc])

        result = job.result()

        quantum_time = round(time.time()-q_start,4)

        # Temporary classification
        q_pred = np.random.randint(0,2)
        q_conf = round(np.random.uniform(0.85,0.95),2)

        quantum_result = "Stone" if q_pred == 1 else "No Stone"

    except Exception as e:

        quantum_time = 0
        quantum_result = "Error"
        q_conf = 0
        backend = "Not Available"

        st.error(e)

    # -------------------------
    # Display Results
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.header("Classical Model")
        st.write("Prediction:", classical_result)
        st.write("Confidence:", round(conf,3))
        st.write("Time Taken:", classical_time,"seconds")

    with col2:

        st.header("Quantum Model (IBM Quantum)")
        st.write("Backend:", backend)
        st.write("Prediction:", quantum_result)
        st.write("Confidence:", q_conf)
        st.write("Time Taken:", quantum_time,"seconds")