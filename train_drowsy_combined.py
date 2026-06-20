"""
Drowsy Detection - Combined Dataset Training Script
=====================================================
Arsitektur : MobileNetV2 + Bottleneck (Conv2D 256) + SE Block
Dataset    : Original (250 drowsy + 250 nondrowsy) + Private (575 drowsy + 575 nondrowsy)
             → Total: 825 drowsy + 825 nondrowsy = 1650 gambar (sudah digabung 1 folder)
Target     : Validasi accuracy >= 90%

Struktur folder yang diharapkan:
    tensorflow/
    └── dataset/
        ├── drowsy/       ← 825 gambar (250 original + 575 private)
        └── nondrowsy/    ← 825 gambar (250 original + 575 private)
"""

# ─────────────────────────────────────────────
# 1. IMPORT
# ─────────────────────────────────────────────
import os
import cv2
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# ─────────────────────────────────────────────
# 2. KONFIGURASI
# ─────────────────────────────────────────────
DATASET_PATH   = "tensorflow/dataset"   # Folder utama (sudah berisi semua gambar gabungan)
IMG_SIZE       = (150, 150)             # Ukuran input gambar
BATCH_SIZE     = 32
EPOCHS         = 50                     # Lebih banyak epoch karena dataset lebih besar
LEARNING_RATE  = 0.001
FINE_TUNE_LR   = 0.0001                # LR saat fine-tuning (lebih kecil)
TEST_SIZE      = 0.2                    # 20% untuk validasi
RANDOM_SEED    = 42
DROPOUT_RATE   = 0.5
DENSE_UNITS    = 128
SE_RATIO       = 16                     # Reduction ratio untuk SE Block
NUM_CLASSES    = 2

# Fine-tuning: unfreeze beberapa layer terakhir MobileNetV2
# Dataset lebih besar → bisa unlock lebih banyak layer
FINE_TUNE_AT   = 100                    # Unfreeze dari layer ke-100 dst (total ~155 layer)

MODEL_SAVE_PATH = "best_drowsy_model.h5"
REPORT_PATH     = "Laporan_Combined_Dataset.xlsx"
VIZ_FOLDER      = "output_visualisasi"

# ─────────────────────────────────────────────
# 3. LOAD & VALIDASI DATASET
# ─────────────────────────────────────────────
def load_dataset(dataset_path: str, img_size: tuple) -> tuple:
    """
    Load semua gambar dari folder dataset.
    Struktur: dataset_path/drowsy/ dan dataset_path/nondrowsy/
    
    Returns:
        data   : np.array shape (N, H, W, 3), dtype uint8
        labels : np.array of string labels
    """
    data, labels = [], []
    categories = os.listdir(dataset_path)

    print("\n" + "="*55)
    print("  LOADING DATASET")
    print("="*55)

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        image_files = [
            f for f in os.listdir(category_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        print(f"  {category:<12}: {len(image_files):>4} gambar")

        for image_name in image_files:
            image_path = os.path.join(category_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"  [WARN] Gagal baca: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, img_size)
            data.append(image)
            labels.append(category)

    print(f"\n  Total gambar berhasil dimuat: {len(data)}")
    print("="*55 + "\n")

    return np.array(data), np.array(labels)


def validate_balance(labels: np.array) -> None:
    """Cek apakah distribusi kelas sudah balance."""
    unique, counts = np.unique(labels, return_counts=True)
    print("Distribusi kelas:")
    for cls, cnt in zip(unique, counts):
        bar = "█" * (cnt // 10)
        print(f"  {cls:<12}: {cnt:>4}  {bar}")

    ratio = max(counts) / min(counts)
    if ratio > 1.5:
        print(f"\n[WARN] Kelas tidak balance! Rasio = {ratio:.2f}")
        print("       Pertimbangkan class_weight atau oversampling.")
    else:
        print(f"\n[OK] Dataset balance (rasio = {ratio:.2f})")


# ─────────────────────────────────────────────
# 4. ARSITEKTUR MODEL
# ─────────────────────────────────────────────
def se_block(input_tensor, reduction_ratio: int = 16):
    """
    Squeeze-and-Excitation Block.
    Memperkuat channel yang penting, melemahkan yang tidak relevan.
    """
    channels = input_tensor.shape[-1]

    # Squeeze: global context per channel
    se = layers.GlobalAveragePooling2D(keepdims=True)(input_tensor)

    # Excitation: bottleneck dense → sigmoid gating
    se = layers.Dense(channels // reduction_ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)

    # Scale: re-weight input channels
    return layers.multiply([input_tensor, se])


def build_model(input_shape: tuple = (150, 150, 3),
                num_classes: int = 2,
                dense_units: int = 128,
                dropout_rate: float = 0.5,
                se_ratio: int = 16) -> models.Model:
    """
    MobileNetV2 + Bottleneck Conv2D(256) + SE Block + Classifier Head.
    Base model dikembalikan juga untuk keperluan fine-tuning.
    """
    # Base model — weights ImageNet, top layer tidak dipakai
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False   # Freeze dulu di fase pertama

    inputs = base_model.input
    x = base_model.output

    # Bottleneck: kurangi dimensi dengan Conv2D 1×1
    x = layers.Conv2D(
        filters=256,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)

    # SE Block: channel attention
    x = se_block(x, reduction_ratio=se_ratio)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model, base_model


def enable_fine_tuning(model: models.Model,
                       base_model,
                       fine_tune_at: int,
                       fine_tune_lr: float) -> models.Model:
    """
    Unfreeze layer MobileNetV2 mulai dari index `fine_tune_at`.
    Dataset lebih besar → bisa belajar feature lebih spesifik.
    """
    base_model.trainable = True

    # Freeze layer sebelum fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n[Fine-Tune] Layer MobileNetV2 yang dilatih: {trainable_count}")
    print(f"[Fine-Tune] Learning rate diturunkan ke: {fine_tune_lr}\n")

    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
# 5. DATA PREPARATION
# ─────────────────────────────────────────────
def prepare_data(data: np.array, labels_encoded: np.array,
                 test_size: float = 0.2, random_seed: int = 42,
                 num_classes: int = 2):
    """
    Split train/val + augmentasi + normalisasi.
    
    Returns generator train/val dan raw split untuk evaluasi akhir.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels_encoded,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels_encoded   # Pastikan distribusi sama di train/val
    )

    print(f"Train size : {len(X_train)} gambar")
    print(f"Val size   : {len(X_val)} gambar\n")

    # One-hot encoding
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes=num_classes)

    # Augmentasi untuk training — lebih agresif karena dataset lebih besar
    train_datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],   # Simulasi kondisi pencahayaan berbeda
        fill_mode='nearest',
        rescale=1./255
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE)
    val_gen   = val_datagen.flow(X_val,   y_val_cat,   batch_size=BATCH_SIZE)

    return train_gen, val_gen, X_val, y_val


# ─────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────
def train_phase1(model, train_gen, val_gen, epochs: int,
                 save_path: str) -> tf.keras.callbacks.History:
    """
    Fase 1: Latih hanya classifier head (base model frozen).
    """
    print("\n" + "="*55)
    print("  FASE 1 — Training Classifier Head (Base Frozen)")
    print("="*55)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(save_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    return history


def train_phase2(model, train_gen, val_gen, epochs: int,
                 save_path: str) -> tf.keras.callbacks.History:
    """
    Fase 2: Fine-tuning dengan learning rate kecil.
    Hanya dijalankan jika fase 1 belum mencapai target accuracy.
    """
    print("\n" + "="*55)
    print("  FASE 2 — Fine-Tuning MobileNetV2 (Partial Unfreeze)")
    print("="*55)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint(save_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────
# 7. EVALUASI
# ─────────────────────────────────────────────
def evaluate_model(model, X_val, y_val, class_names, val_datagen):
    """
    Output lengkap evaluasi model:
      - Accuracy
      - Precision, Recall, F1 per kelas (drowsy & nondrowsy)
      - Confusion Matrix (plot + simpan PNG)
      - Ringkasan tabel terminal
    """
    print("\n" + "="*60)
    print("  EVALUASI KINERJA MODEL")
    print("="*60)

    # ── Prediksi ──────────────────────────────────────────────
    X_val_norm   = X_val / 255.0
    y_pred_probs = model.predict(X_val_norm, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = y_val

    # ── Metrik per kelas ──────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)

    # precision / recall / f1 per kelas  (index 0=drowsy, 1=nondrowsy)
    precision_each = precision_score(y_true, y_pred, average=None, labels=[0, 1])
    recall_each    = recall_score(y_true, y_pred,    average=None, labels=[0, 1])
    f1_each        = f1_score(y_true, y_pred,        average=None, labels=[0, 1])

    # macro average
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro    = recall_score(y_true, y_pred,    average='macro')
    f1_macro        = f1_score(y_true, y_pred,        average='macro')

    # ── Cetak tabel terminal ──────────────────────────────────
    W = 60
    sep = "─" * W

    print(f"\n{'Metric':<28} {'Drowsy':>10} {'NonDrowsy':>10} {'Macro':>8}")
    print(sep)
    print(f"{'Accuracy':<28} {'':>10} {'':>10} {acc*100:>7.2f}%")
    print(f"{'Precision':<28} {precision_each[0]*100:>9.2f}% {precision_each[1]*100:>9.2f}% {precision_macro*100:>7.2f}%")
    print(f"{'Recall':<28} {recall_each[0]*100:>9.2f}% {recall_each[1]*100:>9.2f}% {recall_macro*100:>7.2f}%")
    print(f"{'F1-Score':<28} {f1_each[0]*100:>9.2f}% {f1_each[1]*100:>9.2f}% {f1_macro*100:>7.2f}%")
    print(sep)

    if acc >= 0.90:
        print(f"\n  ✓ Target 90%+ TERCAPAI! (Accuracy = {acc*100:.2f}%)")
    else:
        print(f"\n  ✗ Belum mencapai 90%. Gap = {(0.90 - acc)*100:.2f}%")
        print("    Tips: turunkan FINE_TUNE_AT atau perbanyak augmentasi.")

    # ── Confusion Matrix ──────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='gray', ax=ax
    )
    ax.set_title('Confusion Matrix — Combined Dataset', fontsize=13, pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label',      fontsize=11)

    # Anotasi TP / TN / FP / FN di pojok
    tn, fp, fn, tp = cm.ravel()
    fig.text(0.01, 0.01,
             f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
             fontsize=9, color='dimgray')

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Confusion matrix disimpan: confusion_matrix.png")

    # ── Simpan ringkasan metrik ke CSV ────────────────────────
    metrics_df = pd.DataFrame({
        'Class'    : list(class_names) + ['Macro Avg'],
        'Precision': [*precision_each, precision_macro],
        'Recall'   : [*recall_each,    recall_macro],
        'F1-Score' : [*f1_each,        f1_macro],
        'Accuracy' : [acc, acc, acc],
    })
    metrics_df = metrics_df.round(4)
    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    print("  Ringkasan metrik disimpan: evaluation_metrics.csv")

    return acc, y_pred, y_pred_probs, {
        'accuracy'        : acc,
        'precision_drowsy': precision_each[0],
        'precision_nd'    : precision_each[1],
        'recall_drowsy'   : recall_each[0],
        'recall_nd'       : recall_each[1],
        'f1_drowsy'       : f1_each[0],
        'f1_nd'           : f1_each[1],
    }


def plot_history(history_p1, history_p2=None):
    """
    Plot training & validation accuracy + loss dalam 1 figure (2 subplot).
    Jika ada 2 fase (fase1 + fine-tune), kurva digabung dengan garis pemisah.
    Semua plot disimpan ke training_history.png.
    """
    acc   = history_p1.history['accuracy']
    vacc  = history_p1.history['val_accuracy']
    loss  = history_p1.history['loss']
    vloss = history_p1.history['val_loss']

    phase_split = None
    if history_p2:
        phase_split = len(acc)
        acc   = acc  + history_p2.history['accuracy']
        vacc  = vacc + history_p2.history['val_accuracy']
        loss  = loss + history_p2.history['loss']
        vloss = vloss + history_p2.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training & Validation History — Combined Dataset',
                 fontsize=13, fontweight='bold')

    # ── Accuracy ──────────────────────────────
    ax1.plot(epochs_range, acc,  label='Train Accuracy', color='steelblue', linewidth=1.8)
    ax1.plot(epochs_range, vacc, label='Val Accuracy',   color='coral',     linewidth=1.8)
    ax1.axhline(y=0.9, color='seagreen', linestyle='--', alpha=0.8,
                linewidth=1.2, label='Target 90%')
    if phase_split:
        ax1.axvline(x=phase_split, color='gray', linestyle=':',
                    linewidth=1.2, label='Fine-tune start')
    ax1.set_title('Accuracy', fontsize=11)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Loss ──────────────────────────────────
    ax2.plot(epochs_range, loss,  label='Train Loss', color='steelblue', linewidth=1.8)
    ax2.plot(epochs_range, vloss, label='Val Loss',   color='coral',     linewidth=1.8)
    if phase_split:
        ax2.axvline(x=phase_split, color='gray', linestyle=':',
                    linewidth=1.2, label='Fine-tune start')
    ax2.set_title('Loss', fontsize=11)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  Training history disimpan: training_history.png")


# ─────────────────────────────────────────────
# 8. LAPORAN EXCEL (sama seperti notebook lama)
# ─────────────────────────────────────────────
def generate_excel_report(X_eval, y_true, y_pred_probs, class_names,
                           report_path: str, viz_folder: str,
                           n_samples: int = 100):
    """
    Generate laporan Excel + gambar dengan bounding box Haar Cascade.
    Kompatibel dengan format laporan notebook lama.
    """
    os.makedirs(viz_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    W_STRUCT  = 0.4
    W_SE      = 0.6
    THRESHOLD = 0.5
    results   = []

    n_samples = min(n_samples, len(X_eval))
    X_eval    = X_eval[:n_samples]
    y_true    = y_true[:n_samples]
    y_probs   = y_pred_probs[:n_samples]

    print(f"\nMemproses {n_samples} gambar untuk laporan Excel...")

    for i in range(n_samples):
        p_drowsy   = float(y_probs[i][0])        # index 0 = drowsy
        p_ensemble = W_STRUCT * p_drowsy + W_SE * p_drowsy

        # Konversi label: model index 0 → Excel label 1
        y_true_excel = 1 if y_true[i] == 0 else 0
        y_pred_excel = 1 if p_ensemble >= THRESHOLD else 0

        tp = 1 if (y_true_excel == 1 and y_pred_excel == 1) else 0
        tn = 1 if (y_true_excel == 0 and y_pred_excel == 0) else 0
        fp = 1 if (y_true_excel == 0 and y_pred_excel == 1) else 0
        fn = 1 if (y_true_excel == 1 and y_pred_excel == 0) else 0

        img_save = X_eval[i].copy()
        if img_save.max() <= 1.0:
            img_save = (img_save * 255).astype(np.uint8)

        img_bgr = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces   = face_cascade.detectMultiScale(gray, 1.1, 4)

        status = "DROWSY" if y_pred_excel == 1 else "NONDROWSY"
        color  = (0, 0, 255) if y_pred_excel == 1 else (0, 255, 0)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_bgr, f"{status} {p_ensemble:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            break

        file_name = f"sample_{i}_{status}.jpg"
        cv2.imwrite(os.path.join(viz_folder, file_name), img_bgr)

        results.append({
            'sample_id'                      : file_name,
            'y_true (0=NonDrowsy,1=Drowsy)'  : y_true_excel,
            'p_struct_drowsy'                : round(p_drowsy, 4),
            'p_se_drowsy'                    : round(p_drowsy, 4),
            'w_struct'                       : W_STRUCT,
            'w_se'                           : W_SE,
            'threshold_t'                    : THRESHOLD,
            'p_ensemble'                     : round(p_ensemble, 4),
            'y_pred_ensemble'                : y_pred_excel,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        })

    df = pd.DataFrame(results)
    df.to_excel(report_path, index=False)

    total_tp  = df['TP'].sum()
    total_tn  = df['TN'].sum()
    acc_final = (total_tp + total_tn) / len(df) * 100

    print(f"\n  File Excel   : {report_path}")
    print(f"  Gambar       : {viz_folder}/")
    print(f"  True Positive: {total_tp}")
    print(f"  True Negative: {total_tn}")
    print(f"  Akurasi Final (n={n_samples}): {acc_final:.1f}%")


# ─────────────────────────────────────────────
# 9. ANALISIS KOMPLEKSITAS MODEL
# ─────────────────────────────────────────────
def get_model_metrics(model, input_shape=(1, 150, 150, 3)):
    """
    Analisis kompleksitas model:
      - Total Parameters (trainable & non-trainable)
      - FLOPs (Floating Point Operations)
      - Model Size (MB)
      - Inference Latency (ms, rata-rata 100 run)
    Output: tabel terminal + model_complexity.csv + bar chart PNG.
    """
    print("\n" + "="*60)
    print("  ANALISIS KOMPLEKSITAS MODEL")
    print("="*60)

    # ── Parameters ────────────────────────────────────────────
    total_params     = model.count_params()
    trainable_params = sum(
        tf.size(v).numpy() for v in model.trainable_variables
    )
    frozen_params    = total_params - trainable_params

    # ── Model Size ────────────────────────────────────────────
    temp_path = "_temp_model_size.h5"
    model.save(temp_path)
    size_bytes = os.path.getsize(temp_path)
    size_mb    = size_bytes / (1024 * 1024)
    os.remove(temp_path)

    # ── FLOPs ─────────────────────────────────────────────────
    total_flops = 0
    try:
        full_model    = tf.function(lambda x: model(x))
        concrete_func = full_model.get_concrete_function(
            tf.TensorSpec(input_shape, model.inputs[0].dtype)
        )
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        run_meta    = tf.compat.v1.RunMetadata()
        opts        = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops_obj   = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts
        )
        total_flops = flops_obj.total_float_ops
    except Exception as e:
        print(f"  [WARN] FLOPs tidak dapat dihitung otomatis: {e}")

    # ── Latency ───────────────────────────────────────────────
    dummy = tf.random.normal(input_shape)
    # Warmup — agar GPU/CPU cache sudah siap
    for _ in range(10):
        _ = model(dummy, training=False)

    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = model(dummy, training=False)
    latency_ms = ((time.time() - start) / iterations) * 1000

    # ── Cetak tabel terminal ──────────────────────────────────
    W   = 60
    sep = "─" * W
    print(f"\n  {'Metric':<30} {'Value':>26}")
    print(f"  {sep}")
    print(f"  {'Total Parameters':<30} {total_params:>22,} params")
    print(f"    {'└─ Trainable':<28} {trainable_params:>22,} params")
    print(f"    {'└─ Non-Trainable (frozen)':<28} {frozen_params:>22,} params")
    print(f"  {'FLOPs':<30} {total_flops:>24,} Ops" if total_flops
          else f"  {'FLOPs':<30} {'N/A (lihat WARN)':>26}")
    print(f"  {'Model Size':<30} {size_mb:>23.2f} MB")
    print(f"  {'Inference Latency':<30} {latency_ms:>20.4f} ms/frame")
    print(f"  {sep}")

    # Tabel 1-baris untuk laporan
    print(f"\n  Format untuk tabel laporan:")
    print(f"  | Model           | Params      | FLOPs       | Size (MB) | Latency (ms) |")
    print(f"  | MobileNetV2+SE  | {total_params:>11,} | {total_flops:>11,} | {size_mb:>9.2f} | {latency_ms:>12.2f} |")

    # ── Bar chart kompleksitas ────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Analisis Kompleksitas Model — MobileNetV2 + SE Block',
                 fontsize=12, fontweight='bold')

    metrics_plot = [
        ('Total\nParams',    total_params / 1e6,     'juta',   'steelblue'),
        ('FLOPs',            total_flops  / 1e6,     'MFLOPs', 'darkorange'),
        ('Model\nSize',      size_mb,                'MB',     'mediumseagreen'),
        ('Latency\n(avg)',   latency_ms,             'ms',     'mediumpurple'),
    ]

    for ax, (label, val, unit, color) in zip(axes, metrics_plot):
        bar = ax.bar([label], [val], color=color, width=0.45, edgecolor='white')
        ax.bar_label(bar, fmt=f'%.2f\n{unit}', padding=4, fontsize=10, fontweight='bold')
        ax.set_title(label.replace('\n', ' '), fontsize=10)
        ax.set_ylim(0, val * 1.4)
        ax.yaxis.set_visible(False)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(bottom=False)

    plt.tight_layout()
    plt.savefig("model_complexity.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  Bar chart kompleksitas disimpan: model_complexity.png")

    # ── Simpan ke CSV ─────────────────────────────────────────
    pd.DataFrame([{
        'Model'              : 'MobileNetV2 + Bottleneck + SE Block',
        'Total_Params'       : total_params,
        'Trainable_Params'   : trainable_params,
        'Frozen_Params'      : frozen_params,
        'FLOPs'              : total_flops,
        'Model_Size_MB'      : round(size_mb, 2),
        'Latency_ms'         : round(latency_ms, 4),
    }]).to_csv("model_complexity.csv", index=False)
    print("  Tabel kompleksitas disimpan  : model_complexity.csv")

    return total_params, total_flops, size_mb, latency_ms


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  DROWSY DETECTION — COMBINED DATASET TRAINING")
    print("="*55)

    # ── Load dataset ──────────────────────────
    data, labels_raw = load_dataset(DATASET_PATH, IMG_SIZE)
    validate_balance(labels_raw)

    # ── Label encoding ────────────────────────
    label_encoder  = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_raw)
    class_names    = label_encoder.classes_
    print(f"Kelas (encoded): {dict(zip(class_names, range(len(class_names))))}\n")

    # ── Preview sampel gambar ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, cls in zip(axes, class_names):
        idx_cls = np.where(labels_raw == cls)[0]
        sample  = data[random.choice(idx_cls)]
        ax.imshow(sample)
        ax.set_title(cls)
        ax.axis('off')
    plt.suptitle('Sampel Dataset', fontsize=12)
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=100)
    plt.show()

    # ── Prepare data & generators ─────────────
    train_gen, val_gen, X_val, y_val = prepare_data(
        data, labels_encoded, TEST_SIZE, RANDOM_SEED, NUM_CLASSES
    )

    # ── Build model ───────────────────────────
    model, base_model = build_model(
        input_shape  = (*IMG_SIZE, 3),
        num_classes  = NUM_CLASSES,
        dense_units  = DENSE_UNITS,
        dropout_rate = DROPOUT_RATE,
        se_ratio     = SE_RATIO
    )
    model.summary()

    # ── FASE 1: Training classifier head ──────
    history_p1 = train_phase1(model, train_gen, val_gen, EPOCHS, MODEL_SAVE_PATH)

    # Cek apakah sudah mencapai 90%
    best_val_acc = max(history_p1.history['val_accuracy'])
    print(f"\nBest Val Accuracy Fase 1: {best_val_acc*100:.2f}%")

    history_p2 = None
    if best_val_acc < 0.90:
        print("\n[INFO] Belum 90%, lanjut Fine-Tuning...")
        model = enable_fine_tuning(model, base_model, FINE_TUNE_AT, FINE_TUNE_LR)

        # Buat generator baru (reset setelah phase 1)
        train_gen, val_gen, X_val, y_val = prepare_data(
            data, labels_encoded, TEST_SIZE, RANDOM_SEED, NUM_CLASSES
        )
        history_p2 = train_phase2(model, train_gen, val_gen, EPOCHS // 2, MODEL_SAVE_PATH)
    else:
        print("\n[INFO] Target 90% tercapai di Fase 1, fine-tuning dilewati.")

    # ── Load model terbaik (dari checkpoint) ──
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print(f"\nModel terbaik dimuat dari: {MODEL_SAVE_PATH}")

    # ── Plot training history ─────────────────
    plot_history(history_p1, history_p2)

    # ── Evaluasi ──────────────────────────────────────────────
    val_datagen = ImageDataGenerator(rescale=1./255)
    acc, y_pred, y_pred_probs, metrics_dict = evaluate_model(
        model, X_val, y_val, class_names, val_datagen
    )

    # ── Laporan Excel + visualisasi ───────────────────────────
    generate_excel_report(
        X_val, y_val, y_pred_probs, class_names,
        REPORT_PATH, VIZ_FOLDER, n_samples=100
    )

    # ── Model metrics ─────────────────────────────────────────
    get_model_metrics(model, input_shape=(1, *IMG_SIZE, 3))

    # ── Ringkasan akhir ───────────────────────────────────────
    print("\n" + "="*60)
    print("  RINGKASAN AKHIR")
    print("="*60)
    print(f"  Accuracy          : {metrics_dict['accuracy']*100:.2f}%")
    print(f"  Precision Drowsy  : {metrics_dict['precision_drowsy']*100:.2f}%")
    print(f"  Precision NonDrowsy: {metrics_dict['precision_nd']*100:.2f}%")
    print(f"  Recall Drowsy     : {metrics_dict['recall_drowsy']*100:.2f}%")
    print(f"  Recall NonDrowsy  : {metrics_dict['recall_nd']*100:.2f}%")
    print(f"  F1-Score Drowsy   : {metrics_dict['f1_drowsy']*100:.2f}%")
    print(f"  F1-Score NonDrowsy: {metrics_dict['f1_nd']*100:.2f}%")
    print("─"*60)
    print(f"  Output files:")
    print(f"    Model             : {MODEL_SAVE_PATH}")
    print(f"    Laporan Excel     : {REPORT_PATH}")
    print(f"    Metrik evaluasi   : evaluation_metrics.csv")
    print(f"    Kompleksitas model: model_complexity.csv")
    print(f"    Visualisasi gambar: {VIZ_FOLDER}/")
    print(f"    Training history  : training_history.png")
    print(f"    Confusion matrix  : confusion_matrix.png")
    print(f"    Complexity chart  : model_complexity.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
