"""
01_ml_core_concepts.py — Classical ML workflow, in the vocabulary you need
==========================================================================

WHAT THIS FILE TEACHES
----------------------
• The ML workflow: data → preprocessing → train → evaluate → deploy.
• sklearn: LogisticRegression, train_test_split, accuracy_score, confusion_matrix.
• Feature engineering — what makes a good feature.
• Overfitting vs underfitting.
• Classification vs regression vs clustering.
• Precision, recall, F1 — WHY each metric matters in a real product.

HOW TO RUN
----------
    pip install scikit-learn numpy
    python 01_ml_core_concepts.py

REAL-WORLD SCENARIO
-------------------
Build a spam-email classifier using 4 features:
    - word_count
    - has_link      (binary)
    - is_from_known_sender  (binary)
    - caps_ratio    (0..1)

Show the confusion matrix, explain precision vs recall for spam,
show which metric matters most and why.
"""

import os
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
    )
    from sklearn.cluster import KMeans
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    print("⚠️  scikit-learn not installed. Run:  pip install scikit-learn")

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Build a labeled dataset — realistic spam features
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Building the dataset")
print("=" * 70)

# Feature semantics:
#   [word_count / 100, has_link, is_from_known_sender, caps_ratio]
# Label: 1 = spam, 0 = not spam

rng = np.random.default_rng(seed=42)

# Generate 400 non-spam ("ham") emails: shorter, fewer caps, usually from known senders
ham_X = np.column_stack([
    rng.normal(0.5, 0.2, 400),     # avg 50 words
    rng.random(400) < 0.15,        # 15% have links
    rng.random(400) < 0.8,         # 80% from known senders
    rng.uniform(0.02, 0.10, 400),  # low caps ratio
]).astype(float)
ham_y = np.zeros(400)

# Generate 400 spam emails: longer, more links, unknown senders, more CAPS
spam_X = np.column_stack([
    rng.normal(1.5, 0.4, 400),     # avg 150 words
    rng.random(400) < 0.85,        # 85% have links
    rng.random(400) < 0.10,        # 10% from known senders
    rng.uniform(0.15, 0.45, 400),  # HIGH CAPS RATIO
]).astype(float)
spam_y = np.ones(400)

X = np.vstack([ham_X, spam_X])
y = np.concatenate([ham_y, spam_y])

feature_names = ["word_count/100", "has_link", "known_sender", "caps_ratio"]
print(f"Dataset: {X.shape[0]} emails, {X.shape[1]} features")
print(f"Class balance: {int((y==0).sum())} ham, {int((y==1).sum())} spam")


# ─────────────────────────────────────────────────────────────
# SECTION 2: The universal train/test split
# ─────────────────────────────────────────────────────────────
if _has_sklearn:
    print("\n" + "=" * 70)
    print("SECTION 2: train_test_split")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train:  {len(X_train)}  |  Test:  {len(X_test)}  (stratified — same class ratios)")


    # ─────────────────────────────────────────────────────────────
    # SECTION 3: Train a Logistic Regression classifier
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 3: Train a Logistic Regression")
    print("=" * 70)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Model has learned a weight per feature — negative weights push toward "ham":
    print("Learned coefficients (larger positive = more spammy):")
    for name, coef in sorted(zip(feature_names, model.coef_[0]), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<20} {coef:+.3f}")


    # ─────────────────────────────────────────────────────────────
    # SECTION 4: Evaluate — accuracy, precision, recall, F1
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 4: Evaluation metrics")
    print("=" * 70)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.3f}   (overall correct rate)")
    print(f"Precision: {prec:.3f}  (of flagged spam, how many were really spam)")
    print(f"Recall:    {rec:.3f}   (of real spam, how many we caught)")
    print(f"F1:        {f1:.3f}    (harmonic mean of precision & recall)")


    # ─────────────────────────────────────────────────────────────
    # SECTION 5: Confusion matrix — the diagnostic
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 5: Confusion matrix")
    print("=" * 70)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"""
                     PREDICTED
                    ham    spam
              ┌──────────────────┐
       ham    │  {tn:>5}   {fp:>5}   │  ← FP: false alarms (real email blocked)
    ACTUAL    │                  │
       spam   │  {fn:>5}   {tp:>5}   │  ← FN: missed spam (reaches inbox)
              └──────────────────┘

    True Positives  (spam correctly caught):     {tp}
    True Negatives  (ham correctly delivered):   {tn}
    False Positives (real email blocked):        {fp}  ← usually the WORST error
    False Negatives (spam reached inbox):        {fn}
    """)

    print("""FOR SPAM DETECTION:
    - The COST of an FP (blocking a real email) is HIGH — user loses information.
    - The COST of an FN (spam reaches inbox) is LOW — user hits delete.
    - Therefore: OPTIMIZE FOR PRECISION.
    Compare to fraud detection where FN cost is huge → OPTIMIZE FOR RECALL.
    """)

    print("Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))


    # ─────────────────────────────────────────────────────────────
    # SECTION 6: Overfitting vs Underfitting — visualize with a curve
    # ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SECTION 6: Overfitting vs Underfitting")
    print("=" * 70)

    # A learning curve shows train vs validation score across dataset sizes.
    # Underfit  → both stay LOW.
    # Overfit   → train stays HIGH, validation stays LOW (big gap).
    # Good fit  → both converge HIGH.
    sizes, train_scores, val_scores = learning_curve(
        LogisticRegression(max_iter=1000), X, y, cv=3,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0], scoring="f1"
    )
    print("Train size  |  Train F1  |  Val F1   |  Gap  (want small)")
    for s, tr, vl in zip(sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
        print(f"  {int(s):>7}    |  {tr:>7.3f}  |  {vl:>7.3f}  |  {tr-vl:+.3f}")

    print("""
Interpretation for THIS problem: gap is small → model is well-fit.
If the gap were 0.10+, we'd be overfitting (need more data, regularization, or
simpler features). If both were <0.7 we'd be underfitting (need better features
or a more expressive model).
    """)


    # ─────────────────────────────────────────────────────────────
    # SECTION 7: A tiny unsupervised example — clustering with KMeans
    # ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SECTION 7: Unsupervised — cluster tickets into K=3 groups")
    print("=" * 70)

    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    print(f"Cluster sizes: {np.bincount(labels)}")

    # Cluster purity vs true labels — do our clusters actually separate spam from ham?
    for k in range(3):
        mask = labels == k
        spam_frac = y[mask].mean()
        print(f"  cluster {k}:  size {mask.sum():>3}, spam fraction = {spam_frac:.2%}")

    print("""
Clustering found the spam / ham structure WITHOUT ever seeing labels — this is
the same math used to auto-topic-cluster support tickets before you have labels.
    """)


# ─────────────────────────────────────────────────────────────
# SECTION 8: Regression cameo — predicting resolution time (mock)
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 8: Regression teaser (predict resolution time)")
print("=" * 70)
print("""
Regression is the same workflow with a CONTINUOUS target:
    ticket_features → predicted_resolution_hours (a real number)

Metrics for regression are MAE / MSE / RMSE / R², not P/R/F1.
Same sklearn API:
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
""")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
You now speak the ML dialect fluently:

  • train_test_split (with stratify for classification)
  • Fit → predict → score
  • Accuracy is often MISLEADING — always look at precision/recall/F1
  • Confusion matrix is the truth serum — always inspect it when debugging
  • Different domains optimize different metrics
  • Overfitting = big train/val gap
  • Underfitting = both scores low
  • Unsupervised (KMeans) finds structure without labels

Every LLM-era task reuses this: eval sets, LLM-judge classifiers, RAG scoring,
router models. Same vocabulary.

Next: 02_text_and_tokens.py — tokens, embeddings, cosine, semantic search.
""")
