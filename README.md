# NVIDIA Quantum Hybrid

This repository shows a small classical–quantum–classical workflow with an explicit safety gate. It is written to match the current NVIDIA work on quantum to GPU hybrid computing and to show how AI safety and AI security logic can control access to quantum routines.

Official NVIDIA announcement  
https://nvidianews.nvidia.com/news/nvidia-nvqlink-quantum-gpu-computing

## Project overview

- Classical stage prepares and normalises data.
- A policy checks whether the data stays inside an approved envelope.
- If the policy allows it, the quantum step is executed.
- The pipeline emits structured JSON with policy, backend and latency fields. This is suitable for audit, MLOps and security monitoring.

This is a simple reference that connects current hybrid quantum–GPU news with secure and trustworthy AI concepts.

## Why this is useful

- Shows how to put a safety/policy check in front of a quantum call, which is what you want in secure AI or space/defence contexts.
- Produces JSON that can be logged, audited or sent to an MLOps/SOC pipeline, so it is easy to demo to non-quantum teams.
- Can be swapped from a simulator to CUDA-Q or a partner QPU without changing the classical logic, so it is future-ready.

## Features

- classical → policy → quantum → classical loop
- Qiskit Aer simulator as default backend
- registry of policies (small, strict)
- JSON output with pipeline version, policy name, quantum backend, latency
- ready to swap to CUDA-Q or to a real neutral-atom or photonic backend

## Requirements

```text
numpy
qiskit
qiskit-aer
```

Install

```bash
pip install -r requirements.txt
```

You can also install manually

```bash
pip install numpy qiskit qiskit-aer
```

## Run

```bash
python hybrid_secure_demo.py
```

The script runs two examples. One passes the policy and calls the quantum circuit. The other fails the policy and does not call the quantum circuit.

## File layout

```text
nvidia-quantum-hybrid/
├── requirements.txt
├── hybrid_secure_demo.py
└── README.md
```

- `hybrid_secure_demo.py` is the main demo.
- `requirements.txt` keeps the environment minimal.

## Example output

```json
{
  "pipeline_version": "0.3-nvidia-quantum-hybrid",
  "policy_used": "small",
  "policy_tag": "ok",
  "features_meta": {
    "mean": 1.012,
    "std": 0.04,
    "anomaly_score": 0.26
  },
  "quantum_called": true,
  "quantum_backend": "qiskit_sim",
  "quantum_confidence": 0.462,
  "noise_level": 0.03,
  "reason": "ok",
  "explain": "policy=small, mean=1.012, theta=0.222",
  "latency_s": 0.152
}
```

## Extending

- edit `run_quantum(...)` to call a different backend
- add more policies to the `POLICIES` dictionary
- expose the function as an HTTP service
- add stronger logging instead of `print`
- add provenance or signature checks before the quantum call

## Cite this demo

If you use or adapt this repository, please cite

> Sylvester Kaczmarek (2025). *NVIDIA Quantum Hybrid* (Version 0.2). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

Badge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## License

MIT License

© 2025 Sylvester Kaczmarek
