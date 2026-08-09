import anndata as ad
import numpy as np
import pytest

from scanchor.train import train


def _write_toy_dataset(path, n_cells=10, embed_dim=8):
    rng = np.random.default_rng(0)
    obs = {
        "batch": rng.choice(["b1", "b2"], n_cells),
        "cell_type": rng.choice(["ct1", "ct2"], n_cells),
        "donor_id": rng.choice(["d1", "d2"], n_cells),
        "total_counts": rng.normal(1000, 100, n_cells),
        "pct_counts_mt": rng.normal(5, 1, n_cells),
    }
    adata = ad.AnnData(X=rng.normal(size=(n_cells, 5)), obs=obs)
    adata.obsm["X_scGPT"] = rng.normal(size=(n_cells, embed_dim)).astype("float32")
    adata.write_h5ad(path)


def _base_config(ref_path, checkpoint_out, batch_size):
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
        "model": {"embed_dim": 8, "cat_embed_dim": 4, "covariate_dim": 8, "hidden_dim": 16},
        "training": {
            "batch_size": batch_size,
            "epochs": 1,
            "learning_rate": 1e-3,
            "contrastive_temperature": 0.1,
            "contrastive_weight": 1.0,
            "variance_weight": 1.0,
            "min_variance_ratio": 0.8,
            "checkpoint_out": str(checkpoint_out),
        },
    }


def test_train_raises_instead_of_silently_running_zero_batches(tmp_path):
    ref_path = tmp_path / "ref.h5ad"
    _write_toy_dataset(ref_path, n_cells=10)
    config = _base_config(ref_path, tmp_path / "head.pt", batch_size=64)

    with pytest.raises(ValueError, match="zero minibatches"):
        train(config)


def test_train_runs_when_batch_size_fits_dataset(tmp_path):
    ref_path = tmp_path / "ref.h5ad"
    _write_toy_dataset(ref_path, n_cells=20)
    config = _base_config(ref_path, tmp_path / "head.pt", batch_size=8)

    head = train(config)

    assert head is not None
    assert (tmp_path / "head.pt").exists()
