import csv
from pathlib import Path


# ===Def feedback 
FEEDBACK_LOG_PATH = Path("E:/.csv")




def save_feedback(timestamp, predicted_label, confidence, user_feedback, correct_label):
    """
    Save doctor/user feedback into CSV log file.
   
    Params:
    - timestamp (str): Datetime string
    - predicted_label (str): AI prediction
    - confidence (float): Prediction confidence
    - user_feedback (str): Yes / No / Skip
    - correct_label (str): Optional correct diagnosis if No
   
    Saved columns:
    timestamp, predicted_label, confidence, user_feedback, correct_label
    """
    header = ["timestamp", "predicted_label", "confidence", "user_feedback", "correct_label"]
    row = [timestamp, predicted_label, confidence, user_feedback, correct_label]


    file_exists = FEEDBACK_LOG_PATH.exists()


    with open(FEEDBACK_LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)




# === Example Usage 
if __name__ == "__main__":
    save_feedback("2025-05-03 14:00", "AFIB", 92.5, "Yes", "")
    save_feedback("2025-05-03 14:05", "NSTEMI", 85.1, "No", "STEMI")


