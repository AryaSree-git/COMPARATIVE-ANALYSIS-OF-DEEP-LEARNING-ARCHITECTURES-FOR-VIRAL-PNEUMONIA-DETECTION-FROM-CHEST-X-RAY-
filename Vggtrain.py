import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==================================
# START TIMER
# ==================================

start_time = time.time()

# ==================================
# DATASET PATH
# ==================================

dataset_path = r"Desktop\Lungh phenumonia\Dataset_Split"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# ==================================
# LOAD TRAIN DATASET
# ==================================

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + r"\train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# ==================================
# LOAD VALIDATION DATASET
# ==================================

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + r"\val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# ==================================
# OPTIMIZE DATA LOADING
# ==================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==================================
# DATA AUGMENTATION
# ==================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# ==================================
# LOAD PRETRAINED VGG16
# ==================================

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# ==================================
# BUILD MODEL
# ==================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.vgg16.preprocess_input(x)

x = base_model(x, training=False)

# Better than Flatten()
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# ==================================
# COMPILE MODEL
# ==================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==================================
# MODEL SUMMARY
# ==================================

model.summary()

# ==================================
# CALLBACKS
# ==================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath="best_vgg16_weights.h5",
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# ==================================
# TRAIN MODEL
# ==================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ==================================
# TRAINING TIME
# ==================================

end_time = time.time()

training_time = end_time - start_time

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")

print(f"Training Time : {training_time/60:.2f} Minutes")

# ==================================
# BEST VALIDATION ACCURACY
# ==================================

best_val_acc = max(history.history['val_accuracy'])

print(f"Best Validation Accuracy : {best_val_acc*100:.2f}%")

# ==================================
# ACCURACY GRAPH
# ==================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Training vs Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.grid(True)

plt.show()

# ==================================
# LOSS GRAPH
# ==================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Training vs Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.grid(True)

plt.show()

# ==================================
# SAVE COMPLETE MODEL
# ==================================

try:
    model.save("vgg16_pneumonia_final.h5")
    print("\nFinal model saved successfully.")
except Exception as e:
    print("\nModel save warning:", e)

# ==================================
# FINAL SUMMARY
# ==================================

print("\n===================================")
print("FINAL SUMMARY")
print("===================================")

print(f"Best Validation Accuracy : {best_val_acc*100:.2f}%")
print(f"Training Time            : {training_time/60:.2f} Minutes")

print("\nFiles Generated:")
print("1. best_vgg16_weights.h5")
print("2. vgg16_pneumonia_final.h5")