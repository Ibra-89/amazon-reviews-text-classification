# Amazon Product Reviews - Text Classification

Predicting whether an Amazon product is one customers are **satisfied** with or one that
**needs improvement**, using only the text of its reviews — and finding out *what* people
actually complain about.

A Text Analytics / NLP project built with Python, NLTK and Scikit-Learn.

---

## Problem

A star rating tells you *that* a product is doing poorly, not *why*. Reading thousands of
reviews by hand is not practical. This project uses the review text to automatically sort
products into satisfied vs. needs-improvement, and surfaces the words behind each group.

## Dataset

- **Source:** [Amazon Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
- **Size:** 1,465 products, each combining several customer reviews with one overall rating.
- Download `amazon.csv` from the link above and place it next to the notebook before running.

## Method

1. **Text cleaning** — lowercase, remove links/HTML/numbers/punctuation, drop stop words, lemmatize (NLTK).
2. **Feature extraction** — TF-IDF on the top 3,000 words plus bigrams.
3. **Split** — 80% train / 20% test, stratified.
4. **Model** — Logistic Regression with balanced class weights (the data leans positive).

## Results

- **Accuracy:** ~77% (note: the positive class is ~76%, so recall matters more than accuracy here).
- **Recall on the negative class:** ~0.65 — the model flags about two out of three problem products on its own.
- **Top complaint words:** remote, jar, ear, poor, bud, warm, month, mic, hair, motor
  → issues cluster around audio products, kitchen appliances, and personal-care items, plus
  durability ("warm", "month").
- **Top praise words:** cable, easy, speed, perfect, mouse, excellent, fast, installation.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Amazon_Text_Classification.ipynb
```

Then run all cells (the NLTK data downloads automatically on first run).

## Tech stack

Python · pandas · NLTK · scikit-learn · matplotlib · seaborn

## Limitations

Class imbalance, single product category (electronics/accessories), TF-IDF misses context/sarcasm,
and the text is aggregated at the product level rather than per individual review.
