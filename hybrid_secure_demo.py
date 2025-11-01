import time
import json
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


PIPELINE_VERSION = "0.2-nvidia-quantum-hybrid"


def classical_feature_stage(data):
    """
    Classical preprocessing.
    Normalise, collect basic stats, compute a tiny anomaly score.
    """
    arr = np.array(data, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr) + 1e-8)
    norm = (arr - mean) / std
    anomaly_score = float(np.mean(np.abs(norm)))
    meta = {
        "mean": mean,
        "std": std,
        "anomaly_score": anomaly_score,
    }
    return norm, meta


def policy_envelope_small(norm_features):
    """
    Allow values inside [-2.5, 2.5].
    """
    if np.any(np.abs(norm_features) > 2.5):
        return False, "small_envelope_violation"
    return True, "ok"


def policy_envelope_strict(norm_features):
    """
    More strict envelope.
    """
    if np.any(np.abs(norm_features) > 1.8):
        return False, "strict_envelope_violation"
    return True, "ok"


POLICIES = {
    "small": policy_envelope_small,
    "strict": policy_envelope_strict,
}


def run_policy(norm_features, policy_name="small"):
    policy_fn = POLICIES.get(policy_name, policy_envelope_small)
    return policy_fn(norm_features)


def run_quantum_qiskit(theta):
    """
    Simple 1 qubit circuit on Qiskit Aer.
    """
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sim = AerSimulator()
    job = sim.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts()
    prob_1 = counts.get("1", 0) / 1024.0
    return prob_1


def run_quantum(theta, backend="qiskit_sim"):
    """
    Switchable quantum backend.
    For now we only support Qiskit simulator.
    """
    if backend == "qiskit_sim":
        return run_quantum_qiskit(theta), "qiskit_sim"

    # fallback
    return run_quantum_qiskit(theta), "qiskit_sim"


def log_event(payload):
    """
    Very simple logger.
    Replace with proper logging or webhook later.
    """
    print("[LOG]", json.dumps(payload))


def run_secure_hybrid(sample, policy_name="small", backend="qiskit_sim"):
    """
    Full classical -> policy -> quantum -> classical loop.
    """
    t0 = time.time()

    # 1. classical features
    feats, feat_meta = classical_feature_stage(sample)

    # 2. runtime policy
    safe, tag = run_policy(feats, policy_name=policy_name)

    result = {
        "pipeline_version": PIPELINE_VERSION,
        "policy_used": policy_name,
        "policy_tag": tag,
        "features_meta": feat_meta,
    }

    if not safe:
        # quantum step blocked
        result.update(
            {
                "quantum_called": False,
                "reason": "blocked_by_policy",
            }
        )
        result["latency_s"] = round(time.time() - t0, 4)
        log_event(result)
        return result

    # 3. quantum step allowed
    theta = float(np.clip(np.mean(feats) * np.pi, -np.pi, np.pi))
    prob_1, used_backend = run_quantum(theta, backend=backend)

    # 4. final
    result.update(
        {
            "quantum_called": True,
            "quantum_backend": used_backend,
            "quantum_confidence": prob_1,
            "reason": "ok",
        }
    )
    result["latency_s"] = round(time.time() - t0, 4)

    log_event(result)
    return result


if __name__ == "__main__":
    # sample expected to pass
    sample_ok = [1.0, 1.02, 1.05, 0.98, 1.01]
    out_ok = run_secure_hybrid(sample_ok, policy_name="small")
    print("=== OK sample result ===")
    print(json.dumps(out_ok, indent=2))

    # sample expected to fail
    sample_bad = [1.0, 4.5, 1.05, 0.98, 1.01]
    out_bad = run_secure_hybrid(sample_bad, policy_name="strict")
    print("=== BAD sample result ===")
    print(json.dumps(out_bad, indent=2))
