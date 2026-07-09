import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.keras.applications import DenseNet121
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

dataset_path = r"Dataset_Split"

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
# PERFORMANCE OPTIMIZATION
# ==================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==================================
# DATA AUGMENTATION
# ==================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# ==================================
# LOAD PRETRAINED DENSENET121
# ==================================

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# ==================================
# BUILD MODEL
# ==================================

inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = tf.keras.applications.densenet.preprocess_input(x)

x = base_model(x, training=False)

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
    filepath="best_densenet121_weights.h5",
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

# ==================================
# BEST VALIDATION ACCURACY
# ==================================

best_val_acc = max(history.history['val_accuracy'])

# ==================================
# ACCURACY GRAPH
# ==================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'],
         label='Training Accuracy')

plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy')

plt.title('DenseNet121 Training vs Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.grid(True)

plt.show()

# ==================================
# LOSS GRAPH
# ==================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'],
         label='Training Loss')

plt.plot(history.history['val_loss'],
         label='Validation Loss')

plt.title('DenseNet121 Training vs Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.grid(True)

plt.show()

# ==================================
# SAVE FINAL MODEL
# ==================================

model.save("densenet121_pneumonia_final.h5")

# ==================================
# FINAL SUMMARY
# ==================================

print("\n===================================")
print("DENSENET121 TRAINING COMPLETED")
print("===================================")

print(f"Best Validation Accuracy : {best_val_acc*100:.2f}%")
print(f"Training Time            : {training_time/60:.2f} Minutes")

print("\nFiles Generated:")
print("1. best_densenet121_weights.h5")
print("2. densenet121_pneumonia_final.h5")