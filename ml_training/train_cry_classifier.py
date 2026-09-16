"""
Cry cause classifier training script for TinyTalks — v3.

v2 (merged datasets) revealed a real problem: adding more hungry/
pain_discomfort data without adding tired data worsened the imbalance,
and selecting the "best" classifier by raw validation ACCURACY picked a
model that scored well by never predicting tired at all (0% recall).
This version fixes that by selecting on macro-F1 instead (treats every
class equally, so a classifier can't win by ignoring the minority class),
and gives tired extra augmentation weight to partially offset its
much smaller original sample count.
"""

import os
import numpy as np
import librosa
import torch
from transformers import ASTFeatureExtractor, ASTModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

DONATEACRY_ROOT = r"C:\Users\Nimra Nawaz\Desktop\TINY TALKS\backend\donateacry-corpus-master\donateacry-corpus-master\donateacry_corpus_cleaned_and_updated_data"
INFANTS_CRY_ROOT = r"C:\Users\Nimra Nawaz\Desktop\TINY TALKS\backend\ml_training\infants_cry_sound"

CLASS_SOURCES = {
    "hungry": [
        (DONATEACRY_ROOT, "hungry"),
        (INFANTS_CRY_ROOT, "Hungry"),
    ],
    "tired": [
        (DONATEACRY_ROOT, "tired"),
    ],
    "pain_discomfort": [
        (DONATEACRY_ROOT, "belly_pain"),
        (DONATEACRY_ROOT, "discomfort"),
        (INFANTS_CRY_ROOT, "Uncomfortable"),
    ],
}

# Per-class augmentation multiplier -- tired gets more to partially offset
# its much smaller original sample count relative to the other two classes.
AUGMENT_MULTIPLIER = {
    "hungry": 0,          # already large, no augmentation needed
    "tired": 6,
    "pain_discomfort": 3,
}

SAMPLE_RATE = 16000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Loading pretrained AST model (cached from earlier run)...")
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
embedding_model = ASTModel.from_pretrained(MODEL_NAME)
embedding_model.eval()


def get_embedding(y, sr):
    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def load_audio(filepath, augment=None):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    if augment == "pitch":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))
    elif augment == "stretch":
        rate = np.random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)
    elif augment == "noise":
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise
    return y, sr


def gather_filepaths():
    filepaths, labels = [], []
    class_names = list(CLASS_SOURCES.keys())

    for label_idx, class_name in enumerate(class_names):
        class_count = 0
        for root, subfolder in CLASS_SOURCES[class_name]:
            folder_path = os.path.join(root, subfolder)
            if not os.path.isdir(folder_path):
                print(f"  WARNING: folder not found, skipping: {folder_path}")
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(".wav"):
                    filepaths.append(os.path.join(folder_path, fname))
                    labels.append(label_idx)
                    class_count += 1
        print(f"{class_name}: {class_count} total original .wav clips (all sources combined)")

    return filepaths, np.array(labels), class_names


def build_split(filepaths, labels, class_names, split_name):
    X, y_out = [], []
    total = len(filepaths)

    for i, (fp, label) in enumerate(zip(filepaths, labels)):
        if i % 20 == 0:
            print(f"  [{split_name}] {i}/{total} files embedded...")
        try:
            y, sr = load_audio(fp)
            X.append(get_embedding(y, sr))
            y_out.append(label)
        except Exception as e:
            print(f"  Skipping {fp}: {e}")
            continue

        class_name = class_names[label]
        multiplier = AUGMENT_MULTIPLIER.get(class_name, 0)
        if multiplier > 0:
            augment_types = ["pitch", "stretch", "noise"]
            for _ in range(multiplier):
                aug = np.random.choice(augment_types)
                try:
                    y_aug, sr = load_audio(fp, augment=aug)
                    X.append(get_embedding(y_aug, sr))
                    y_out.append(label)
                except Exception as e:
                    print(f"  Skipping augmented {fp}: {e}")

    return np.array(X), np.array(y_out)


def main():
    print("Gathering original file list from ALL sources...")
    filepaths, labels, class_names = gather_filepaths()
    filepaths = np.array(filepaths)
    print(f"\nTotal original files across all sources: {len(filepaths)}")

    print("\nSplitting ORIGINAL FILES into train/val/test...")
    fp_train, fp_temp, y_train_orig, y_temp_orig = train_test_split(
        filepaths, labels, test_size=0.3, random_state=RANDOM_SEED, stratify=labels
    )
    fp_val, fp_test, y_val_orig, y_test_orig = train_test_split(
        fp_temp, y_temp_orig, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp_orig
    )
    print(f"Original files -> Train: {len(fp_train)}  Val: {len(fp_val)}  Test: {len(fp_test)}")

    print("\nExtracting embeddings for TRAIN split...")
    X_train, y_train = build_split(fp_train, y_train_orig, class_names, "train")
    print("Extracting embeddings for VAL split...")
    X_val, y_val = build_split(fp_val, y_val_orig, class_names, "val")
    print("Extracting embeddings for TEST split...")
    X_test, y_test = build_split(fp_test, y_test_orig, class_names, "test")

    print(f"\nAfter augmentation -> Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Embedding dimension: {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED
        ),
    }

    # Select by MACRO-F1, not accuracy -- macro-F1 treats every class
    # equally regardless of size, so a classifier can't win by silently
    # ignoring the minority class the way it did last run.
    best_name, best_clf, best_val_macro_f1 = None, None, -1
    for name, clf in candidates.items():
        clf.fit(X_train_scaled, y_train)
        val_pred = clf.predict(X_val_scaled)
        val_acc = clf.score(X_val_scaled, y_val)
        val_macro_f1 = f1_score(y_val, val_pred, average="macro")
        tired_idx = class_names.index("tired")
        tired_f1 = f1_score(y_val, val_pred, labels=[tired_idx], average="macro")
        print(f"{name}: val accuracy={val_acc:.3f}  macro-F1={val_macro_f1:.3f}  tired-F1={tired_f1:.3f}")
        if val_macro_f1 > best_val_macro_f1:
            best_name, best_clf, best_val_macro_f1 = name, clf, val_macro_f1

    print(f"\nBest classifier (by macro-F1): {best_name} (macro-F1 {best_val_macro_f1:.3f})")

    print("\n=== FINAL TEST SET EVALUATION (leakage-free, merged datasets, macro-F1 selection) ===")
    y_pred = best_clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(best_clf, "saved_model/cry_cause_classifier.joblib")
    joblib.dump(scaler, "saved_model/scaler.joblib")
    print(f"\nSaved classifier ({best_name}) and scaler to saved_model/")


if __name__ == "__main__":
    main()