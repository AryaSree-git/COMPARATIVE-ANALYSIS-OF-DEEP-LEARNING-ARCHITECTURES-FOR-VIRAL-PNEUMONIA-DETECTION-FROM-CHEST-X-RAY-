import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

# =====================================
# PATHS
# =====================================

MODEL_PATH = "densenet121_pneumonia_final.h5"

TEST_PATH = r"Dataset_Split"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model(MODEL_PATH)

print("DenseNet121 Model Loaded Successfully")

# =====================================
# LOAD TEST DATASET
# =====================================

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)

# =====================================
# GET TRUE LABELS
# =====================================

y_true = np.concatenate(
    [labels for images, labels in test_ds],
    axis=0
)

y_true = y_true.flatten()

# =====================================
# PREDICTIONS
# =====================================

y_prob = model.predict(test_ds)

y_prob = y_prob.flatten()

y_pred = (y_prob > 0.5).astype(int)

# =====================================
# CALCULATE METRICS
# =====================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

roc_auc = roc_auc_score(y_true, y_prob)

# =====================================
# DISPLAY RESULTS
# =====================================

print("\n===================================")
print("DENSENET121 TEST RESULTS")
print("===================================")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix\n")
print(cm)

# =====================================
# CONFUSION MATRIX HEATMAP
# =====================================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix - DenseNet121")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================
# ROC CURVE
# =====================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(7,5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - DenseNet121")

plt.legend()

plt.grid(True)

plt.show()

# =====================================
# PRECISION-RECALL CURVE
# =====================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(7,5))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve - DenseNet121")

plt.grid(True)

plt.show()

# =====================================
# PERFORMANCE BAR CHART
# =====================================

metrics_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

metrics_values = [
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
]

plt.figure(figsize=(8,5))

bars = plt.bar(
    metrics_names,
    metrics_values
)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha='center'
    )

plt.ylim(0, 1.1)

plt.ylabel("Score")

plt.title("DenseNet121 Performance Metrics")

plt.grid(axis='y')

plt.show()

# =====================================
# FINAL SUMMARY
# =====================================

print("\n===================================")
print("FINAL SUMMARY")
print("===================================")

print(f"Test Accuracy : {accuracy*100:.2f}%")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"ROC-AUC       : {roc_auc:.4f}")

print("\nTesting Completed Successfully")