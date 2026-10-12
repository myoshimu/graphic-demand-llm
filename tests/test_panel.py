from __future__ import annotations

import pandas as pd
import pytest

from src.panel.personas import load_config, quota, sample_personas
from src.panel.stimuli import build_stimulus


def test_quota_sums_and_tracks_weights() -> None:
    c = quota([0.5, 0.3, 0.2], 50)
    assert c.sum() == 50 and list(c) == [25, 15, 10]


def test_personas_match_marginals_and_are_deterministic() -> None:
    cfg = load_config()
    a, b = sample_personas(50, cfg), sample_personas(50, cfg)
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]
    spec = cfg["attributes"]["gender"]
    n_f = sum(p.gender == "female" for p in a)
    assert abs(n_f - 50 * spec["weights"][0]) < 1
    assert all(p.describe_ja() for p in a)


ROW = pd.Series({"class_id": "000000001", "color_id": "106", "item_name": "Sample Short Sleeve Tee",
                 "ip_name": "SampleBrand", "is_original": False, "price_list": 3500,
                 "image_path": "data/cache/images/x.jpg", "image_sha256": "abc"})


def test_masked_condition_hides_names() -> None:
    s = build_stimulus(ROW, "image_masked_text")
    assert "SampleBrand" not in s.text and "3,500" in s.text and s.image_path is not None


def test_text_only_has_no_image() -> None:
    s = build_stimulus(ROW, "text_only")
    assert s.image_path is None and s.image_sha256 is None and "SampleBrand" in s.text


def test_image_only_has_no_text_metadata() -> None:
    s = build_stimulus(ROW, "image_only")
    assert "SampleBrand" not in s.text and "円" not in s.text


def test_artwork_only_not_yet_available() -> None:
    with pytest.raises(NotImplementedError):
        build_stimulus(ROW, "artwork_only")
