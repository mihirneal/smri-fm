from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "aabc_ood_eval" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_aabc_tables import (  # noqa: E402
    add_age_bins,
    clean_synthseg_region,
    parse_age,
    parse_id_event,
    select_representative_subset,
)
from run_brainage_transfer import align_test_features  # noqa: E402


def test_parse_id_event_and_age_policy() -> None:
    assert parse_id_event("HCA6002236_V3") == ("sub-HCA6002236", "V3")
    assert parse_id_event("bad") == (None, None)
    assert parse_age("86") == 86
    assert np.isnan(parse_age("90 or older"))


def test_clean_synthseg_region_matches_master_style() -> None:
    assert clean_synthseg_region("total intracranial") == "total_intracranial"
    assert clean_synthseg_region("brain-stem") == "brain_stem"
    assert clean_synthseg_region("left ventral DC") == "left_ventral_DC"
    assert clean_synthseg_region("ctx-lh-bankssts") == "ctx_lh_bankssts"


def test_representative_subset_keeps_v1_v2_v3() -> None:
    rows = []
    for idx in range(90):
        wave = ["V1", "V2", "V3"][idx % 3]
        rows.append(
            {
                "subject_id": f"sub-HCA{idx:07d}",
                "wave": wave,
                "AgeMRI": 40 + idx % 45,
                "Sex": ["F", "M"][idx % 2],
                "site": ["WashU", "UMinn", "MGH"][idx % 3],
                "scanner": ["A", "B", "C"][idx % 3],
                "source_study": ["HCA", "AABC"][idx % 2],
            }
        )
    df = pd.DataFrame(rows)
    df["age_bin"] = add_age_bins(df)
    subset = select_representative_subset(df, n=30, seed=7)

    assert len(subset) == 30
    assert set(subset["wave"]) == {"V1", "V2", "V3"}
    assert subset["subset_index"].tolist() == list(range(1, 31))


def test_align_test_features_uses_train_columns() -> None:
    train = pd.DataFrame({"a": [1.0], "b": [2.0]})
    test = pd.DataFrame({"b": [3.0], "c": [4.0]})
    aligned = align_test_features(train, test)

    assert aligned.columns.tolist() == ["a", "b"]
    assert np.isnan(aligned.loc[0, "a"])
    assert aligned.loc[0, "b"] == 3.0
