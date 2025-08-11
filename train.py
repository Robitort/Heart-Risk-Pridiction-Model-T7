import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os


from data_loader import load_ptbxl_data, create_tf_dataset
from model import build_multimodal_advanced_model
from config import DEFAULT_FS, BATCH_SIZE, EPOCHS, SEED, MODEL_DIR
from utils_logger import get_logger


# === Setup 
logger = get_logger(log_file=MODEL_DIR / "training.log")
tf.random.set_seed(SEED)
np.random.seed(SEED)


# === Load Data 
logger.info("Loading PTB-XL data...")
X, C, Y, class_names = load_ptbxl_data()


logger.info(f"Dataset shape: ECG={X.shape}, Clinical={C.shape}, Labels={Y.shape}")
num_classes = Y.shape[1]


# === Train/Validation Split 
X_train, X_val, C_train, C_val, Y_train, Y_val = train_test_split(
    X, C, Y, test_size=0.2, random_state=SEED, stratify=Y
)


# === Build Datasets 
train_ds = create_tf_dataset(X_train, C_train, Y_train, batch_size=BATCH_SIZE)
val_ds = create_tf_dataset(X_val, C_val, Y_val, batch_size=BATCH_SIZE, shuffle=False)


# === Build Model 
logger.info("Building model...")
model = build_multimodal_advanced_model(
    input_shape=X.shape[1:],  # (5000, 12)
    clin_dim=C.shape[1],    # typically 2
    num_classes=num_classes
)
model.summary(print_fn=logger.info)


# === Callbacks 
checkpoint_path = MODEL_DIR / "best_model.h5"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR / "tensorboard_logs")
]


# === Train ===
logger.info("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


logger.info("Training complete. Best model saved to: %s", checkpoint_path)
