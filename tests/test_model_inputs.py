from __future__ import annotations

import pandas as pd

from scripts.train_gpu_return_model import load_training_frame


def test_gpu_training_frame_joins_embeddings_by_month(tmp_path):
    panel_dir = tmp_path / "data" / "processed" / "panel"
    panel_dir.mkdir(parents=True)
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    pd.DataFrame(
        {
            "date": dates,
            "permno": [1, 1, 1],
            "secid": [10, 10, 10],
            "next_ret": [0.01, 0.02, -0.01],
            "mean_iv": [0.20, 0.25, 0.30],
        }
    ).to_parquet(panel_dir / "surface_characteristic_state_ibes_regsho_external_panel.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "permno": [1, 1, 1],
            "secid": [10, 10, 10],
            "surface_ae_01": [0.1, 0.2, 0.3],
        }
    ).to_parquet(panel_dir / "surface_autoencoder_embeddings.parquet", index=False)

    frame, features, _path = load_training_frame(tmp_path, None, min_nonmissing=0.50)

    assert len(frame) == 3
    assert "surface_ae_01" in features
