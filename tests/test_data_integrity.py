"""Subject-level rules on the local ADNI-derived tables (skipped when the data is not here).

Hard rules: no subject in more than one split or in both cohorts; the E1-E5 features, BAG table
and splits describe the same MCI subjects and scans; MCI test subjects never reach training
outputs; the test-set lock holds. Only sets and counts are compared; nothing is printed.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import data_path


def test_no_subject_in_two_splits():
    for name in ("cn_splits.csv", "mci_splits.csv"):
        s = pd.read_csv(data_path(f"data/tables/{name}"))
        assert not s["RID"].duplicated().any(), name
        assert set(s["split"]) == {"train", "val", "test"}, name


def test_no_subject_in_both_cohorts():
    cn = pd.read_csv(data_path("data/tables/cn_splits.csv"))
    mci = pd.read_csv(data_path("data/tables/mci_splits.csv"))
    assert not set(cn["RID"]) & set(mci["RID"])


def test_cn_training_scans_only_from_training_subjects():
    scans = pd.read_csv(data_path("data/tables/cn_train_scans.csv"))
    splits = pd.read_csv(data_path("data/tables/cn_splits.csv"))
    train = set(splits.loc[splits["split"] == "train", "RID"])
    assert set(scans["RID"]) <= train


def test_mci_features_match_bag_table_and_splits():
    idx = pd.read_csv(data_path("data/features/mci_voxels_4mm_index.csv"))
    bag = pd.read_csv(data_path("outputs/brain_age/bag/bag_table.csv"))
    splits = pd.read_csv(data_path("data/tables/mci_splits.csv"))
    X = np.load(data_path("data/features/mci_voxels_4mm.npy"), mmap_mode="r")
    assert X.shape[0] == len(idx) and not idx["RID"].duplicated().any()
    a = idx.sort_values("RID").reset_index(drop=True)
    b = bag[["RID", "image_id", "split"]].sort_values("RID").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    merged = idx.merge(splits, on="RID", suffixes=("", "_table"))
    assert len(merged) == len(idx) and (merged["split"] == merged["split_table"]).all()


def test_e1_e2_used_same_subjects_and_folds():
    o1 = pd.read_csv(data_path("outputs/progression/E1/oof_train.csv"))
    o2 = pd.read_csv(data_path("outputs/progression/E2/oof_train.csv"))
    pd.testing.assert_frame_equal(o1[["RID", "label", "fold"]], o2[["RID", "label", "fold"]])
    v1 = pd.read_csv(data_path("outputs/progression/E1/val_predictions.csv"))
    v2 = pd.read_csv(data_path("outputs/progression/E2/val_predictions.csv"))
    pd.testing.assert_frame_equal(v1[["RID", "label"]], v2[["RID", "label"]])


def test_mci_test_subjects_never_in_training_outputs():
    splits = pd.read_csv(data_path("data/tables/mci_splits.csv"))
    test = set(splits.loc[splits["split"] == "test", "RID"])
    for exp in ("E1", "E2"):
        o = pd.read_csv(data_path(f"outputs/progression/{exp}/oof_train.csv"))
        v = pd.read_csv(data_path(f"outputs/progression/{exp}/val_predictions.csv"))
        assert not test & set(o["RID"]) and not test & set(v["RID"])


def test_test_set_lock_refuses_second_run(monkeypatch):
    data_path("outputs/progression/test/test_metrics.json")      # the test set was used once
    from src.progression import evaluate
    for argv in (["evaluate", "--set", "test"],
                 ["evaluate", "--set", "test", "--confirm-test"]):
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as e:
            evaluate.main()
        assert "test" in str(e.value)
