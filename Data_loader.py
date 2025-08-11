import os
import numpy as np
import pandas as pd
import wfdb
import ast
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer




from config import DEFAULT_FS, DATASET_DIR, LOG_FILE
from utils_logger import get_logger




logger = get_logger(log_file=LOG_FILE)




def load_ptbxl_data(ptbxl_path=DATASET_DIR, fs=DEFAULT_FS):
    """
    Load PTB-XL ECG signal data and metadata.
    Returns: X (ECG signals), C (clinical features), Y (labels), class_names
    """
    meta_path = ptbxl_path / "ptbxl_metadata.csv"
    meta = pd.read_csv(meta_path)




    filename_col = "filename_lr" if fs == 100 else "filename_hr"
    records_file = ptbxl_path / "RECORDS"
    valid_records = set(open(records_file, "r").read().splitlines())




    signals, clin_feats, label_list = [], [], []
    skipped = 0




    logger.info(f"[DATA] Loading metadata from {meta_path}. Found {len(meta)} records.")




    for _, row in meta.iterrows():
        if filename_col not in row or pd.isna(row[filename_col]):
            skipped += 1
            continue




        fname = row[filename_col]
        if fname not in valid_records:
            logger.warning(f"Skipping unknown file: {fname}")
            skipped += 1
            continue




        rec_base = ptbxl_path / fname
        dat_file = rec_base.with_suffix(".dat")
        hea_file = rec_base.with_suffix(".hea")




        if not (dat_file.exists() and hea_file.exists()):
            logger.warning(f"Missing file(s): {dat_file} or {hea_file}")
            skipped += 1
            continue




        try:
            record = wfdb.rdrecord(str(rec_base))
            sig = record.p_signal.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to read record {rec_base}", exc_info=e)
            skipped += 1
            continue




        if sig is None or sig.ndim != 2 or sig.shape[1] != 12:
            logger.warning(f"Invalid signal shape in {fname}: {sig.shape if sig is not None else 'None'}")
            skipped += 1
            continue




        target_len = 10 * fs
        sig = sig[:target_len, :] if sig.shape[0] >= target_len else np.pad(
            sig, ((0, target_len - sig.shape[0]), (0, 0)), mode="constant"
        )
        sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)
        signals.append(sig)




        # Clinical features
        age = row.get("age", np.nan)
        sex = row.get("sex", "Unknown")
        sex_val = 1.0 if str(sex).upper() == "F" else 0.0
        clin_feats.append([float(age) if pd.notna(age) else 0.0, sex_val])




        # Diagnostic labels
        try:
            scp = row.get("scp_codes", "{}")
            labels = list(ast.literal_eval(scp).keys())
        except Exception:
            labels = []
        label_list.append(labels)




    if not signals:
        raise ValueError("No valid ECG records found.")




    logger.info(f"[DATA] Successfully loaded {len(signals)} signals. Skipped {skipped} entries.")




    X = np.stack(signals)
    C = np.array(clin_feats)




    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_list)
    class_names = mlb.classes_




    return X, C, Y, class_names








def create_tf_dataset(X, C, Y, batch_size=32, shuffle=True):
    """
    Create a batched and prefetched tf.data.Dataset from numpy arrays
    """
    dataset = tf.data.Dataset.from_tensor_slices(((X, C), Y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), seed=42)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
