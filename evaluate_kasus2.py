"""
evaluate_kasus2.py
===================
Implementasi "Kasus 2": mengukur secara empiris apakah input terkurasi
(Skenario B - Curated Guided Essay, akumulasi multi-topik) menghasilkan
prediksi MBTI yang LEBIH AKURAT dibandingkan esai tunggal biasa
(Skenario A - Single Free Essay), menggunakan model & vectorizer yang
sudah dilatih (mbti_models.joblib, tfidf_vectorizer.joblib).

Skrip ini adalah "proses perhitungan hasil" yang sebelumnya belum ada
pada sistem PsychoNLP: sistem sudah bisa memprediksi satu input, tetapi
belum pernah membuktikan secara kuantitatif apakah mode Guided (Skenario
B) benar-benar mengungguli mode Free-Essay (Skenario A).

Metodologi
----------
1. Memuat ulang dataset/mbti_2.csv dan melakukan SPLIT USER-LEVEL yang
   IDENTIK dengan training.py (random_state=42, test_size=0.2), sehingga
   evaluasi ini murni dilakukan pada data uji (test users) yang belum
   pernah dilihat model selama pelatihan -> tidak ada data leakage.
2. Setiap user pada test set difragmentasi menjadi beberapa chunk teks
   ~256 kata (identik dengan training.py) -- chunk-chunk ini berperan
   sebagai proksi dari beberapa "jawaban topik" yang berbeda dari user
   yang sama (menggantikan data topik berlabel yang tidak tersedia
   secara publik).
3. Untuk setiap user yang memiliki >= 2 chunk valid:
   - Skenario A (Single Free Essay): hanya CHUNK PERTAMA (~256 kata)
     yang digunakan sebagai input -> mensimulasikan pengguna yang cuma
     menulis satu esai bebas tanpa dipandu.
   - Skenario B (Curated Guided Essay): chunk-chunk DIAKUMULASI secara
     berurutan (menirukan alur guided multi-topik pada app.js /
     submitGuided) sampai salah satu syarat terpenuhi:
         a. Rerata confidence axis >= CONFIDENCE_TARGET, ATAU
         b. Jumlah kata kumulatif >= WORD_COUNT_THRESHOLD, ATAU
         c. Jumlah langkah (chunk) yang dipakai >= MAX_GUIDED_STEPS
     Ini adalah realisasi kuantitatif dari mekanisme akumulasi kata dan
     penghentian dinamis berbasis confidence yang dijelaskan pada
     PKB.docx (poin 1 & 2).
4. Prediksi dihasilkan memakai model & vectorizer yang SAMA persis untuk
   kedua skenario (tidak ada perbedaan model, murni perbedaan jumlah &
   kurasi teks input) supaya perbandingan adil (apple-to-apple).
5. Metrik yang dihitung untuk kedua skenario, pada himpunan user yang
   SAMA persis (paired comparison, bukan sampel berbeda):
     - Accuracy, F1-score, AUC-ROC per axis (I/E, N/S, F/T, J/P)
     - Exact-match accuracy tipe 4-huruf penuh (mis. "INFJ" vs "INFJ")
     - Rata-rata confidence per axis & confidence keseluruhan
     - Rata-rata jumlah kata & jumlah langkah yang dipakai Skenario B
6. Menghasilkan "Information Gain": delta AUC-ROC / Accuracy
   (Skenario B - Skenario A), sesuai saran laporan pada PKB.docx.

Output:
   - static/plots/kasus2_information_gain.png  (grafik perbandingan)
   - static/plots/kasus2_results.csv           (angka mentah per axis)
   - static/plots/kasus2_summary.json          (ringkasan untuk laporan)
"""

import csv
import json
import os
import re
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Config (identik / selaras dengan training.py & app.py, plus parameter baru
# dari instruksi ekspansi pada PKB.docx)
# --------------------------------------------------------------------------
DATASET_PATH = "dataset/mbti_2.csv"
MODEL_PATH = "mbti_models.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
PLOTS_DIR = "static/plots"

CHUNK_WORD_TARGET = 256          # identik training.py
MIN_CHUNK_WORDS = 10             # identik training.py / app.py

CONFIDENCE_TARGET = 0.55         # target rerata confidence axis (skala 0-1) untuk berhenti lebih awal
WORD_COUNT_THRESHOLD = 750       # ambang kata kumulatif (poin 2 PKB.docx: 750-800 kata)
MAX_GUIDED_STEPS = 3             # identik guidedStep < 3 pada app.js

RANDOM_STATE = 42
TEST_SIZE = 0.2

TRAITS = ["I/E", "N/S", "F/T", "J/P"]
TRAIT_LABELS = {
    "I/E": ("Introvert", "Extrovert"),
    "N/S": ("Intuitive", "Sensing"),
    "F/T": ("Feeling", "Thinking"),
    "J/P": ("Judging", "Perceiving"),
}
LETTER_POS = {"I/E": "I", "N/S": "N", "F/T": "F", "J/P": "J"}
LETTER_NEG = {"I/E": "E", "N/S": "S", "F/T": "T", "J/P": "P"}

# Perluasan MASK_REGEX (poin 4 PKB.docx): tidak hanya 16 tipe MBTI, tapi
# juga fungsi kognitif Jungian & notasi Enneagram yang bisa menyebabkan
# target leakage / self-reference bias.
MBTI_TYPES = [
    "infj", "enfp", "intj", "entp", "intp", "infp", "entj", "enfj",
    "isfj", "istj", "isfp", "istp", "estp", "esfp", "estj", "esfj",
]
COGNITIVE_FUNCTIONS = ["ni", "ne", "si", "se", "ti", "te", "fi", "fe"]
ENNEAGRAM_TERMS = [
    "enneagram", r"\d w\d", r"\dw\d", "tritype", "instinctual variant",
    "sx", "sp so", "so sx",
]
MBTI_JARGON_WORDS = ["mbti", "myers briggs", "myers-briggs", "cognitive function", "cognitive functions"]

_type_pattern = r"\b(" + "|".join(MBTI_TYPES) + r")s?\b"
_cogfunc_pattern = r"\b(" + "|".join(COGNITIVE_FUNCTIONS) + r")\b"
_enneagram_pattern = r"\b(" + "|".join(ENNEAGRAM_TERMS) + r")\b"
_jargon_pattern = r"\b(" + "|".join(re.escape(w) for w in MBTI_JARGON_WORDS) + r")\b"

MASK_REGEX = re.compile(_type_pattern, re.IGNORECASE)
MASK_REGEX_EXPANDED = re.compile(
    "|".join([_type_pattern, _cogfunc_pattern, _enneagram_pattern, _jargon_pattern]),
    re.IGNORECASE,
)


def clean_and_mask_text(text, expanded_mask=True):
    """Identik dengan training.py, dengan opsi masking yang diperluas (poin 4)."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    pattern = MASK_REGEX_EXPANDED if expanded_mask else MASK_REGEX
    text = pattern.sub("[MASK]", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def chunk_text(posts, target_word_count=CHUNK_WORD_TARGET):
    """Identik dengan training.py chunk_text()."""
    chunks = []
    current_chunk = []
    current_words = 0
    for post in posts:
        cleaned_post = clean_and_mask_text(post)
        if not cleaned_post:
            continue
        post_words = cleaned_post.split()
        post_word_count = len(post_words)
        if current_words + post_word_count > target_word_count and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = post_words
            current_words = post_word_count
        else:
            current_chunk.extend(post_words)
            current_words += post_word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def load_test_users():
    """Memuat dataset & mereproduksi split user-level IDENTIK dengan training.py,
    lalu mengembalikan daftar (chunks, labels) HANYA untuk test users."""
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset tidak ditemukan di {DATASET_PATH}")
        sys.exit(1)

    users_data = []
    with open(DATASET_PATH, mode="r", encoding="utf-8") as f:
        csv.field_size_limit(2147483647)
        reader = csv.DictReader(f)
        for row in reader:
            posts = row["text"].split("|||")
            labels = {trait: int(row[trait]) for trait in TRAITS}
            users_data.append((posts, labels))

    _, test_users = train_test_split(users_data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    processed = []
    for posts, labels in test_users:
        chunks = chunk_text(posts, target_word_count=CHUNK_WORD_TARGET)
        chunks = [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]
        if chunks:
            processed.append({"chunks": chunks, "labels": labels})
    return processed


def predict_axis_probs(text, vectorizer, models):
    """Mengembalikan dict trait -> P(class=1) memakai model & vectorizer terlatih."""
    features = vectorizer.transform([text])
    probs = {}
    for trait in TRAITS:
        p1 = models[trait].predict_proba(features)[0][1]
        probs[trait] = p1
    return probs


def axis_confidences(probs):
    return {trait: abs(p - 0.5) * 2 for trait, p in probs.items()}


def build_scenario_a(user):
    """Skenario A: Single Free Essay -> hanya chunk pertama."""
    text = user["chunks"][0]
    return text, len(text.split()), 1


def build_scenario_b(user, vectorizer, models):
    """Skenario B: Curated Guided Essay -> akumulasi chunk berurutan sampai
    confidence target / word threshold / max steps tercapai (mensimulasikan
    alur dinamis pada app.js submitGuided + rekomendasi poin 1 & 2 PKB.docx)."""
    accumulated = []
    for step, chunk in enumerate(user["chunks"][:MAX_GUIDED_STEPS], start=1):
        accumulated.append(chunk)
        combined_text = " ".join(accumulated)
        word_count = len(combined_text.split())
        probs = predict_axis_probs(combined_text, vectorizer, models)
        mean_conf = np.mean(list(axis_confidences(probs).values()))
        if mean_conf >= CONFIDENCE_TARGET or word_count >= WORD_COUNT_THRESHOLD or step >= MAX_GUIDED_STEPS:
            return combined_text, word_count, step
    combined_text = " ".join(accumulated)
    return combined_text, len(combined_text.split()), len(accumulated)


def main():
    print("=" * 70)
    print("KASUS 2: Evaluasi Skenario A (Single Essay) vs Skenario B (Curated Guided)")
    print("=" * 70)

    print("\n[1/4] Memuat model & vectorizer terlatih...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    models = joblib.load(MODEL_PATH)

    print("[2/4] Mereproduksi split user-level test set (identik training.py)...")
    test_users = load_test_users()
    eligible_users = [u for u in test_users if len(u["chunks"]) >= 2]
    print(f"   Total test users             : {len(test_users)}")
    print(f"   User dgn >=2 chunk (eligible) : {len(eligible_users)}  (dipakai utk perbandingan A vs B)")
    print(f"   User dgn hanya 1 chunk        : {len(test_users) - len(eligible_users)}  (dikecualikan, B == A)")

    print("[3/4] Menjalankan prediksi untuk kedua skenario pada setiap user...")
    y_true = {trait: [] for trait in TRAITS}
    probs_a = {trait: [] for trait in TRAITS}
    probs_b = {trait: [] for trait in TRAITS}
    word_counts_a, word_counts_b, steps_b = [], [], []
    exact_match_a, exact_match_b = 0, 0

    for user in eligible_users:
        text_a, wc_a, _ = build_scenario_a(user)
        text_b, wc_b, n_steps = build_scenario_b(user, vectorizer, models)

        pa = predict_axis_probs(text_a, vectorizer, models)
        pb = predict_axis_probs(text_b, vectorizer, models)

        word_counts_a.append(wc_a)
        word_counts_b.append(wc_b)
        steps_b.append(n_steps)

        pred_type_a, pred_type_b, true_type = "", "", ""
        for trait in TRAITS:
            y_true[trait].append(user["labels"][trait])
            probs_a[trait].append(pa[trait])
            probs_b[trait].append(pb[trait])

            pred_type_a += LETTER_POS[trait] if pa[trait] >= 0.5 else LETTER_NEG[trait]
            pred_type_b += LETTER_POS[trait] if pb[trait] >= 0.5 else LETTER_NEG[trait]
            true_type += LETTER_POS[trait] if user["labels"][trait] == 1 else LETTER_NEG[trait]

        if pred_type_a == true_type:
            exact_match_a += 1
        if pred_type_b == true_type:
            exact_match_b += 1

    n = len(eligible_users)
    print(f"   Selesai memproses {n} user.")

    print("[4/4] Menghitung metrik & Information Gain Analysis...")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    auc_a_list, auc_b_list = [], []
    for trait in TRAITS:
        yt = np.array(y_true[trait])
        pa = np.array(probs_a[trait])
        pb = np.array(probs_b[trait])
        preds_a = (pa >= 0.5).astype(int)
        preds_b = (pb >= 0.5).astype(int)

        acc_a = accuracy_score(yt, preds_a)
        acc_b = accuracy_score(yt, preds_b)
        f1_a = f1_score(yt, preds_a)
        f1_b = f1_score(yt, preds_b)
        auc_a = roc_auc_score(yt, pa)
        auc_b = roc_auc_score(yt, pb)
        conf_a = float(np.mean(np.abs(pa - 0.5) * 2))
        conf_b = float(np.mean(np.abs(pb - 0.5) * 2))

        auc_a_list.append(auc_a)
        auc_b_list.append(auc_b)

        rows.append({
            "axis": trait,
            "n_users": n,
            "accuracy_A": round(acc_a, 4), "accuracy_B": round(acc_b, 4), "accuracy_gain": round(acc_b - acc_a, 4),
            "f1_A": round(f1_a, 4), "f1_B": round(f1_b, 4), "f1_gain": round(f1_b - f1_a, 4),
            "auc_roc_A": round(auc_a, 4), "auc_roc_B": round(auc_b, 4), "auc_roc_gain": round(auc_b - auc_a, 4),
            "mean_confidence_A": round(conf_a, 4), "mean_confidence_B": round(conf_b, 4),
        })

        print(f"\n  Axis {trait} ({TRAIT_LABELS[trait][0]} vs {TRAIT_LABELS[trait][1]}):")
        print(f"    Accuracy : A={acc_a:.4f}  B={acc_b:.4f}  (gain={acc_b - acc_a:+.4f})")
        print(f"    F1       : A={f1_a:.4f}  B={f1_b:.4f}  (gain={f1_b - f1_a:+.4f})")
        print(f"    AUC-ROC  : A={auc_a:.4f}  B={auc_b:.4f}  (gain={auc_b - auc_a:+.4f})")
        print(f"    Confidence(mean): A={conf_a:.4f}  B={conf_b:.4f}")

    overall_acc_a = exact_match_a / n
    overall_acc_b = exact_match_b / n
    print(f"\n  EXACT-MATCH 4-LETTER TYPE ACCURACY: A={overall_acc_a:.4f}  B={overall_acc_b:.4f}  (gain={overall_acc_b - overall_acc_a:+.4f})")
    print(f"  Rata-rata kata  -> A={np.mean(word_counts_a):.1f}  B={np.mean(word_counts_b):.1f}")
    print(f"  Rata-rata langkah Skenario B (guided steps) -> {np.mean(steps_b):.2f}")

    # ---- CSV export ----
    csv_path = os.path.join(PLOTS_DIR, "kasus2_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Disimpan: {csv_path}")

    # ---- JSON summary export (untuk draf laporan) ----
    summary = {
        "n_eligible_users": n,
        "n_total_test_users": len(test_users),
        "config": {
            "confidence_target": CONFIDENCE_TARGET,
            "word_count_threshold": WORD_COUNT_THRESHOLD,
            "max_guided_steps": MAX_GUIDED_STEPS,
            "chunk_word_target": CHUNK_WORD_TARGET,
        },
        "mean_word_count_A": float(np.mean(word_counts_a)),
        "mean_word_count_B": float(np.mean(word_counts_b)),
        "mean_guided_steps_B": float(np.mean(steps_b)),
        "exact_match_accuracy_A": overall_acc_a,
        "exact_match_accuracy_B": overall_acc_b,
        "exact_match_gain": overall_acc_b - overall_acc_a,
        "per_axis": rows,
        "mean_auc_roc_A": float(np.mean(auc_a_list)),
        "mean_auc_roc_B": float(np.mean(auc_b_list)),
        "mean_auc_roc_gain": float(np.mean(auc_b_list) - np.mean(auc_a_list)),
    }
    json_path = os.path.join(PLOTS_DIR, "kasus2_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Disimpan: {json_path}")

    # ---- Plot: AUC-ROC comparison bar chart ----
    x = np.arange(len(TRAITS))
    width = 0.35
    axes[0].bar(x - width / 2, auc_a_list, width, label="Skenario A (Single Essay)", color="#f43f5e", edgecolor="black")
    axes[0].bar(x + width / 2, auc_b_list, width, label="Skenario B (Curated Guided)", color="#06b6d4", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TRAITS)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("Information Gain Analysis: AUC-ROC per Axis", fontweight="bold")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].legend()
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # ---- Plot: Exact-match & mean accuracy comparison ----
    acc_a_list = [r["accuracy_A"] for r in rows]
    acc_b_list = [r["accuracy_B"] for r in rows]
    labels2 = TRAITS + ["4-Letter\nExact Match"]
    a_vals = acc_a_list + [overall_acc_a]
    b_vals = acc_b_list + [overall_acc_b]
    x2 = np.arange(len(labels2))
    axes[1].bar(x2 - width / 2, a_vals, width, label="Skenario A (Single Essay)", color="#f43f5e", edgecolor="black")
    axes[1].bar(x2 + width / 2, b_vals, width, label="Skenario B (Curated Guided)", color="#06b6d4", edgecolor="black")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Accuracy per Axis & Exact-Match Tipe 4-Huruf", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"Kasus 2 — Information Gain: Curated Guided (B) vs Single Essay (A)  |  n={n} test users",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plot_path = os.path.join(PLOTS_DIR, "kasus2_information_gain.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Disimpan: {plot_path}")

    print("\nSelesai. Gunakan static/plots/kasus2_summary.json untuk mengisi bagian")
    print("'Information Gain Analysis' pada laporan penelitian Anda.")


if __name__ == "__main__":
    main()
