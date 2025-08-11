from pathlib import Path


# === Paths 
ROOT_DIR = Path("E:/)
DATASET_DIR = Path("E:/ptbxl/PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
MODEL_DIR = ROOT_DIR / "SAVED"
LOG_FILE = ROOT_DIR / "logs" / "ecg_system.log"


# === Dataloader 
DEFAULT_FS = 500  # Switch to 100 if needed


BATCH_SIZE = 32
EPOCHS = 30
SEED = 42
