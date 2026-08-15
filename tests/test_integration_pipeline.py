"""End-to-end integration test: train() through both evaluate protocols on
synthetic data. The other tests in this directory each cover one piece in
isolation (losses, model, single-call train()) -- this one checks the whole
pipeline a real user actually runs (train -> leave-one-batch-out ->
same-donor-across-batch) doesn't break when wired together, using the exact
public entry points (train(), leave_one_batch_out.run(),
replicate_test.run()) rather than internal calls.
"""

import anndata as ad
import numpy as np

from scanchor.evaluate import leave_one_batch_out, replicate_test
from scanchor.train import train


def _write_dataset(path, batches, n_donors=4, cells_per_donor_per_batch=5, embed_dim=8, seed=0):
    """Donors crossed across every listed batch, so donor_consistency_loss
    has real cross-batch pairs to learn from -- mirrors the real reference
    panels this project trains on (Levy, scIB tasks), not a degenerate case.
    """
    rng = np.random.default_rng(seed)
    donors = [f"d{i}" for i in range(n_donors)]
    n_cells = len(batches) * n_donors * cells_per_donor_per_batch

    batch_col, donor_col = [], []
    for b in batches:
        for d in donors:
            batch_col += [b] * cells_per_donor_per_batch
            donor_col += [d] * cells_per_donor_per_batch

    obs = {
        "batch": batch_col,
        "donor_id": donor_col,
        "cell_type": rng.choice(["ct1", "ct2"], n_cells),
        "total_counts": rng.normal(1000, 100, n_cells),
        "pct_counts_mt": rng.normal(5, 1, n_cells),
    }
    adata = ad.AnnData(X=rng.normal(size=(n_cells, 5)), obs=obs)
    adata.obsm["X_scGPT"] = rng.normal(size=(n_cells, embed_dim)).astype("float32")
    adata.write_h5ad(path)
    return n_cells


def _config(ref_path, held_out_path, checkpoint_out, embed_dim=8):
    return {
        "reference_panel": {
            "paths": [str(ref_path)],
            "embedding_key": "X_scGPT",
            "cell_type_col": "cell_type",
            "batch_col": "batch",
            "donor_col": "donor_id",
            "categorical_covariate_cols": ["batch"],
            "continuous_covariate_cols": ["total_counts", "pct_counts_mt"],
        },
        "model": {"embed_dim": embed_dim, "cat_embed_dim": 4, "covariate_dim": 8, "hidden_dim": 16},
        "training": {
            "seed": 0,
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 1e-3,
            "contrastive_temperature": 0.1,
            "contrastive_weight": 1.0,
            "variance_weight": 1.0,
            "donor_weight": 1.0,
            "mmd_weight": 20.0,
            "min_variance_ratio": 0.8,
            "checkpoint_out": str(checkpoint_out),
        },
        "validation": {
            "leave_one_batch_out_path": str(held_out_path),
            "replicate_dataset_path": str(ref_path),
            "donor_col": "donor_id",
        },
    }


def test_train_then_leave_one_batch_out_and_replicate_test(tmp_path):
    ref_path = tmp_path / "reference.h5ad"
    held_out_path = tmp_path / "heldout.h5ad"
    checkpoint_out = tmp_path / "head.pt"

    _write_dataset(ref_path, batches=["b1", "b2"])
    # A batch never seen during training -- the actual inductive-
    # generalization claim under test, same setup as the real benchmark
    # scripts (run_stephenson_benchmark.py, run_scib_benchmark.py).
    _write_dataset(held_out_path, batches=["b3"], seed=1)

    config = _config(ref_path, held_out_path, checkpoint_out)

    head = train(config)
    assert head is not None
    assert checkpoint_out.exists()

    lobo_result = leave_one_batch_out.run(config, checkpoint_out)
    for key in (
        "batch_mixing_purity_before",
        "batch_mixing_purity_after",
        "label_knn_purity_before",
        "label_knn_purity_after",
    ):
        assert key in lobo_result
        assert 0.0 <= lobo_result[key] <= 1.0

    replicate_result = replicate_test.run(config, checkpoint_out)
    for key in ("donor_retrieval_accuracy_before", "donor_retrieval_accuracy_after"):
        assert key in replicate_result
        assert 0.0 <= replicate_result[key] <= 1.0
