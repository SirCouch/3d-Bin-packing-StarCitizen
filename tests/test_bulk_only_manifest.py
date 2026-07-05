import random

from src.scu_manifest_generator import (
    BULK_ONLY_RUN_TYPE,
    FILLER_CONTAINERS,
    generate_scu_manifest,
    is_bulk_only_manifest_target,
)


def test_hull_c_is_bulk_only_by_name_not_total_volume():
    normal_large = [(12, 6, 8), (8, 8, 4)]
    hull_c_scale = [(24, 24, 8)] * 8
    many_small_grids = [(8, 8, 4)] * 6

    assert is_bulk_only_manifest_target(normal_large, ship_name="C2 Hercules") is False
    assert is_bulk_only_manifest_target(normal_large, ship_name="Hull-c") is True
    assert is_bulk_only_manifest_target(hull_c_scale) is False
    assert is_bulk_only_manifest_target(many_small_grids) is False


def test_bulk_only_manifest_excludes_filler_containers():
    random.seed(1234)
    hull_c_scale = [(24, 24, 8)] * 8

    manifest = generate_scu_manifest(
        grids_list=hull_c_scale,
        target_fill_ratio=0.8,
        difficulty="hard",
        run_type=BULK_ONLY_RUN_TYPE,
    )

    assert manifest
    assert all(entry["scu_type"] not in FILLER_CONTAINERS for entry in manifest)
    assert {entry["scu_type"] for entry in manifest} <= {"8 SCU", "16 SCU", "24 SCU", "32 SCU"}
