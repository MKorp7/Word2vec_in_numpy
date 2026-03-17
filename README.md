# Word2vec_in_numpy
Skip-gram word2vec with negative sampling implemented from scratch in pure NumPy 

---

## Model overview

**Skip-gram** predicts surrounding context words given a centre word. For each training pair (centre *c*, context *o*), the **negative sampling** objective is:

```
L = −log σ(v_c · u_o) − Σₖ log σ(−v_c · u_nk)
```

where `u_nk` are noise words drawn from a smoothed unigram distribution (`freq^0.75`). This approximates the full softmax at a fraction of the cost.

The gradients are in the following forms:

```
e_pos = σ(v_c · u_o) − 1          ∈ (−1, 0]
e_neg = σ(v_c · u_nk)             ∈  (0, 1)

∂L/∂v_c   = e_pos · u_o  +  Σₖ e_neg_k · u_nk
∂L/∂u_o   = e_pos · v_c
∂L/∂u_nk  = e_neg_k · v_c
```

Two embedding matrices are maintained — `W_in` (centre word lookup) and `W_out` (context/noise lookup) and averaged at inference time.

---

## Implementation details

### Sparse Adam

Standard Adam accumulates moment estimates for every embedding row on every step, even rows with zero gradient. For a 30k-word vocabulary where a typical batch touches ~5k rows, this means 25k wasted updates per batch and incorrect bias-correction factors for rows that haven't actually been seen.

The sparse variant here only updates `m`, `v`, and `params` for rows that appear in the current batch. Duplicate row indices within a batch are summed before the moment update.

### Dynamic windowing

For each centre word, the actual window radius `w` is sampled uniformly from `[1, window]`. This gives closer words a higher effective weight since they appear in more windows on average.

### Subsampling

Frequent but uninformative words (*the*, *of*, *and*) are dropped from the corpus with probability:

```
P_drop(w) = 1 − √(t / f(w))
```

where `f(w)` is the word's corpus frequency and `t = 1e-5` is the threshold. 

---

## Ablation study

Run before the full training loop. Each configuration changes exactly one hyperparameter from the baseline, so the effects are isolated.

| Config | D / win / K | loss | cos(king,queen) | cos(paris,berlin) |
|---|---|---|---|---|
| D=50 (fewer dims) | 50 / 5 / 5 | 2.5181 | 0.913 | −0.234 |
| **D=100 ← baseline** | **100 / 5 / 5** | **2.4966** | **0.889** | **0.572** |
| D=200 (more dims) | 200 / 5 / 5 | 2.4496 | 0.796 | 0.313 |
| window=2 (narrow) | 100 / 2 / 5 | 2.5234 | 0.913 | −0.243 |
| window=10 (wide) | 100 / 10 / 5 | 2.4293 | 0.845 | 0.175 |
| K=1 (few negatives) | 100 / 5 / 1 | 1.235 | 0.769 | 0.54 |
| **K=15 (many negatives)** | **100 / 5 / 15** | **3.527** | **0.942** | **0.857** |

**Takeaways:**
- **K=15 clearly wins** on both semantic probes. The higher loss is expected as more negative terms means a larger raw loss value; the scale is not comparable across K values.
- **D=200 underperforms on the slice** because the larger parameter space needs more data to converge. Likely worth revisiting on the full corpus.
- **window=5 is the right balance** — wider windows blur syntactic precision in exchange for topical coherence, which hurts the capital-city probe.

The final config uses `k_negatives=15` based on these results.

---

## Saving and loading

Two save formats are provided:

```python
# Full checkpoint: saves weights, Adam state, vocab, config, loss history.
# Training can be resumed from this.
model.save_checkpoint("model.npz")

# Word2vec text format: final averaged vectors, gensim-compatible.
# Smaller, universally readable, but training cannot be resumed.
model.save_embeddings_txt("embeddings.txt")
```

Loading:

```python
# Reconstruct full model, ready for inference or continued training
model = SkipGramNS.load_checkpoint("model.npz")
model.most_similar("king")

# Load into gensim
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format("embeddings.txt", binary=False)
wv.most_similar("king")
```

The checkpoint stores the Adam step counter `t` so bias correction remains correct if you resume training mid-run.

---

## Requirements

```
numpy
tqdm
matplotlib
scikit-learn   # for t-SNE visualisation
scipy          # for WordSim-353 Spearman ρ
```

No GPU required. 

---


## File structure

```
word2vec.ipynb      — full implementation and walkthrough
model.npz           — trained checkpoint (after running)
embeddings.txt      — word2vec text format export (after running)
loss.png            — training loss curve
tsne.png            — t-SNE projection of top-300 embeddings
```

---

## References

- Mikolov et al. (2013) — [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Mikolov et al. (2013) — [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- text8 corpus — http://mattmahoney.net/dc/text8.zip
