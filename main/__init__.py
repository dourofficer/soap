"""soap/main — the simplified, self-contained SOAP runner.

Predict the decisive-error step of a failed multi-agent trajectory from a frozen proxy
model's internals:

    v_t          mean-pooled hidden states over step t's own tokens, encoded in context
    S(s_t)       = 1 / (mean_{c in C} <v_t, V[:,c]>^2 + eps)          higher = error
    w_{i,t}      fraction of step t's query attention landing in predecessor step i
    S~(s_i)      = S(s_i) + gamma * (sum_{t>i} w_{i,t} S(s_t)) / (sum_{t>i} w_{i,t})
    t_hat        = argmax_i S~(s_i)

SIX CHOICES ARE FROZEN IN CODE HERE, not exposed as config:

    pooling      mean          (both poolings are still EXTRACTED; only scoring pins it)
    SVD          uncentered
    scorer       proj
    orientation  inverse, folded into S so ranking is always descending
    score_norm   none
    weighted     false

``src/`` keeps every one of those axes implemented and sweepable; this package trades
that generality for a pipeline you can read in one sitting. What remains swept:

    base      position x c_begin x c_end
    rescore   layer_range x gamma x w x strategy

Seeds are NOT swept either: one 3-seed triple per subset is declared in
``configs-main/<ds>.yaml`` and every backbone uses it, so a reported number is the mean
over those three splits.

    python -m main extract   --config configs-main/ww.yaml
    python -m main sweep     --config configs-main/ww.yaml
    python -m main select    --config configs-main/ww.yaml
    python -m main reproduce --config configs-main/ww.yaml --row backprop

This package imports NOTHING from ``src/`` (enforced by tests/test_main.py).
"""
__version__ = "1.0.0"
