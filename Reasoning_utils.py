  import numpy as np




def lead_contribution_explain(ecg_signal):
    """
    Simple lead contribution estimator based on signal energy.
    (Surrogate for actual SHAP in this version).


    Params:
    - ecg_signal (np.array): (5000, 12)


    Returns:
    - List of tuples: [(lead_name, contribution_score), ...]
    """
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    energy_per_lead = np.sum(ecg_signal ** 2, axis=0)
    total_energy = np.sum(energy_per_lead)
    contrib_percent = (energy_per_lead / total_energy) * 100
    report = list(zip(leads, np.round(contrib_percent, 2)))
    return sorted(report, key=lambda x: x[1], reverse=True)  # highest first




def generate_reasoning_report(pred_labels, confidence, threshold_flag, lead_report, risk_scores, ecg_quality):
    """
    Generate doctor-style AI reasoning summary.


    Params:
    - pred_labels (list): Predicted labels
    - confidence (float): Confidence %
    - threshold_flag (str): Threshold comment
    - lead_report (list): From lead_contribution_explain()
    - risk_scores (dict): Risk scores
    - ecg_quality (str): ECG quality grade


    Returns:
    - reasoning (str)
    """
    lead_top = lead_report[:3]
    lead_str = ", ".join([f"{l} ({s}%)" for l, s in lead_top])


    risks_str = ", ".join([f"{k}: {v}%" for k, v in risk_scores.items()])


    diag = pred_labels[0] if pred_labels else "No major condition detected"


    reasoning = f"Diagnosis: {diag}\nConfidence: {confidence:.1f}% ({threshold_flag})\nTop Leads: {lead_str}\nRisks: {risks_str}\nECG Quality: {ecg_quality}"


    return reasoning




# === Example Usage 
if __name__ == "__main__":
    dummy_ecg = np.random.normal(0, 0.2, (5000, 12))
    leads = lead_contribution_explain(dummy_ecg)
    reason = generate_reasoning_report(["AFIB"], 92.5, "✅ Above threshold", leads, {"ASCVD (%)": 12, "Framingham (%)": 8}, "Good")
    print(reason)
