import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# DATASET PATH
# =========================

dataset_path = r"C:\Users\ARYA SREE ASHOK\OneDrive\Desktop\Lungh phenumonia\Dataset_Split"

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 15

# =========================
# LOAD TRAIN DATA
# =========================

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + r"\train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# =========================
# LOAD VALIDATION DATA
# =========================

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + r"\val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# =========================
# PERFORMANCE
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# =========================
# EFFICIENTNET-B0
# =========================

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# MODEL
# =========================

inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# =========================
# COMPILE
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_efficientnet_b0.keras",
    monitor='val_accuracy',
    save_best_only=True
)

# =========================
# TRAIN
# =========================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# =========================
# TRAINING TIME
# =========================

end_time = time.time()

training_time = end_time - start_time

print("\nTraining Completed")
print(f"Training Time: {training_time/60:.2f} Minutes")

# =========================
# BEST VALIDATION ACCURACY
# =========================

best_val_acc = max(history.history['val_accuracy'])

print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])

plt.show()

# =========================
# LOSS GRAPH
# =========================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Validation'])

plt.show()

# =========================
# SAVE FINAL MODEL
# =========================

model.save("efficientnet_b0_pneumonia.keras")

print("Model Saved Successfully")