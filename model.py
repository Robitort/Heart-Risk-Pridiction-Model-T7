from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, Dense, Dropout, Reshape, GlobalAveragePooling2D,
    Multiply, Concatenate, BatchNormalization, Add, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.activations import swish








def ChannelAttention():
   
    def layer(x):
        avg_pool = GlobalAveragePooling2D()(x)
        dense1 = Dense(x.shape[-1] // 2, activation='relu')(avg_pool)
        dense2 = Dense(x.shape[-1], activation='sigmoid')(dense1)
        scale = Multiply()([x, Reshape((1, 1, x.shape[-1]))(dense2)])
        return scale
    return layer








def ResidualConvBlock(x, filters, kernel_size=(3, 1), block_name="res_block"):
    """
    A simple 2-layer Conv2D residual block
    """
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', activation='swish', name=f"{block_name}_conv1")(x)
    x = BatchNormalization(name=f"{block_name}_bn1")(x)
    x = Conv2D(filters, kernel_size, padding='same', activation='swish', name=f"{block_name}_conv2")(x)
    x = BatchNormalization(name=f"{block_name}_bn2")(x)




    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=(1, 1), padding='same', name=f"{block_name}_proj")(shortcut)




    x = Add(name=f"{block_name}_add")([x, shortcut])
    return x








def build_multimodal_advanced_model(input_shape=(5000, 12), clin_dim=2, num_classes=5):
    """
    Build an advanced multimodal ECG model with residual blocks, multi-scale 2D conv, attention, and clinical fusion.
    """




    # === ECG Branch ===
    ecg_input = Input(shape=input_shape, name="ecg_input")
    x = Conv1D(64, 11, activation='swish', padding='same')(ecg_input)
    x = Reshape((input_shape[0], input_shape[1], 1))(x)  # -> (5000, 12, 1)




    # Multi-scale branches
    branch1 = Conv2D(32, (50, 1), activation='swish', padding='same', name="branch1")(x)
    branch2 = Conv2D(32, (100, 1), activation='swish', padding='same', name="branch2")(x)
    branch3 = Conv2D(32, (150, 1), activation='swish', padding='same', name="branch3")(x)




    merged = Concatenate(name="concat_branches")([branch1, branch2, branch3])




    # Residual block
    x = ResidualConvBlock(merged, filters=96, kernel_size=(3, 1), block_name="res_block1")




    # Channel attention
    x = ChannelAttention()(x)




    # Global Pooling
    x = GlobalAveragePooling2D(name="ecg_pool")(x)




    # === Clinical Data ===
    clin_input = Input(shape=(clin_dim,), name="clinical_input")
    y = Dense(64, activation='relu')(clin_input)
    y = BatchNormalization()(y)




    # === Fusion ===
    z = Concatenate(name="fusion")([x, y])
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.3)(z)
    output = Dense(num_classes, activation='sigmoid', name="diagnosis")(z)




    model = Model(inputs=[ecg_input, clin_input], outputs=output)
    model.compile(
        optimizer=Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", AUC(name="auc")]
    )




    return model
