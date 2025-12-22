from pathlib import Path

from scripts.main_easy import build_argparser, run


def test_easy_smoke(tmp_path: Path):
    p = build_argparser()
    args = p.parse_args(
        [
            "--audio_dir",
            "dataset/audio",
            "--out_dir",
            str(tmp_path),
            "--epochs",
            "1",
            "--max_items",
            "8",
            "--n_clusters",
            "2",
            "--latent_dim",
            "4",
            "--embed",
            "tsne",
        ]
    )
    out = run(args)
    assert (tmp_path / "metrics_easy.csv").exists()
    assert isinstance(out["metrics"], list)
