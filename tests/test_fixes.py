"""六项缺陷修复的回归测试。

原 ``test_bracket_hdr_deghost.py`` 只验证“函数不返回 None”；本文件逐项
验证描述中的六个修复确实成立，并覆盖解码屏障 / 全缓存回滚 / 色调映射 /
审计与确定性等对外约定。全部场景确定性生成，不依赖外部照片或网络。
"""

from __future__ import annotations

import numpy as np
import pytest

import imaging.bracket_hdr_deghost as mod


ALL_FNS = [
    "local_align", "ghost_mask", "ev_vignette_correct", "decode_barrier",
    "rollback_all", "tone_map_out", "load_bracket", "align", "merge",
    "compose", "validate_ghost", "metrics", "audit", "fallback",
    "determinism",
]


# ---------------------------------------------------------------------------
# 0. 基本契约 / 标量自检
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_FNS)
def test_scalar_contract_non_none(name):
    fn = getattr(mod, name)
    assert fn(1) is not None


def test_every_scalar_self_check_reports_ok():
    for name in ALL_FNS:
        res = getattr(mod, name)(1)
        if isinstance(res, dict) and "ok" in res:
            assert res["ok"] is not False, f"{name}(1) 自检失败: {res.get('error')}"


def test_audit_matches_external_smoke_shape():
    # 与同类任务外部验证脚本相同的取值方式。
    r = mod.audit(1)
    summary = {k: r[k] for k in
               ("n_frames", "valid", "tonemap_monotonic", "deterministic", "digest")}
    assert summary["n_frames"] == 3
    assert summary["valid"] is True
    assert summary["tonemap_monotonic"] is True
    assert summary["deterministic"] is True
    assert isinstance(summary["digest"], str) and len(summary["digest"]) >= 16


# ---------------------------------------------------------------------------
# 1. 按参考帧做局部对齐（局部视差 / 边缘建筑重影）
# ---------------------------------------------------------------------------

def test_local_alignment_beats_global_on_edge_parallax():
    ds = mod.make_parallax_stack()
    rep = mod.local_align(ds["frames"], ds["reference"])
    assert rep["ok"]
    g = rep["edge_residual_global"]
    loc = rep["edge_residual_local"]
    assert g > 0.02                      # 纯全局对齐在边缘确实有残差
    assert loc < 0.6 * g                 # 局部对齐把边缘残差压到 60% 以下
    assert rep["improves_edges"] is True
    assert len(rep["aligned"]) == 3
    assert rep["aligned"][0].shape == ds["frames"][0].shape


def test_align_entry_reports_homography_and_local_refinement():
    rep = mod.align(1)
    assert rep["ok"]
    assert rep["local_refined"] is True
    assert rep["homography"].shape == (3, 3)
    assert rep["edge_residual_local"] < rep["edge_residual_global"]


def test_local_align_accepts_single_stack_argument():
    ds = mod.make_parallax_stack()
    rep = mod.local_align(ds["frames"])
    assert rep["ok"] and len(rep["aligned"]) == 3


def test_local_flow_handles_rgb():
    ds = mod.make_parallax_stack(rgb=True)
    rep = mod.local_align(ds["frames"], 1)
    assert rep["ok"]
    assert rep["aligned"][0].ndim == 3 and rep["aligned"][0].shape[2] == 3


# ---------------------------------------------------------------------------
# 2. 鬼影检测：亮度差掩码 + 掩码感知合并
# ---------------------------------------------------------------------------

def test_ghost_mask_uses_luminance_difference():
    rep = mod.ghost_mask(1)
    assert rep["ok"]
    assert rep["method"] == "luminance_difference_mask"
    assert len(rep["brightness_diff"]) == 3
    assert rep["combined"].ndim == 2
    assert 0.0 < rep["ghost_ratio"] < 0.6


def test_ghost_mask_hits_overlap_and_not_clean_corners():
    ds = mod.make_ghost_stack()
    rep = mod.ghost_mask(ds["frames"][1], ds["frames"])
    combined = rep["combined"].astype(bool)
    assert combined[ds["overlap_roi"]].mean() > 0.8
    assert combined[:4, :4].mean() < 0.05
    assert combined[-4:, -4:].mean() < 0.05


def test_deghost_merge_removes_overlap_ghost_that_median_keeps():
    ds = mod.make_ghost_stack()
    m = mod.merge(ds["frames"], reference_index=1,
                  align_first=False, correct_ev=False)
    assert m["ok"]
    roi = ds["overlap_roi"]
    gt = ds["background"][roi]
    deghost_err = float(np.mean(np.abs(mod._lum(m["merged"])[roi] - gt)))
    median_err = float(np.mean(np.abs(mod._lum(m["median"])[roi] - gt)))
    assert median_err > 0.2              # 中值在两帧重叠处确实留下鬼影
    assert deghost_err < 0.05            # 掩码感知合并回到参考干净背景
    assert deghost_err < 0.3 * median_err


def test_validate_ghost_reports_improvement():
    ds = mod.make_ghost_stack()
    m = mod.merge(ds["frames"], reference_index=1,
                  align_first=False, correct_ev=False)
    v = mod.validate_ghost(m)
    assert v["ok"]
    assert v["ghost_removed_vs_median"] is True
    assert v["residual_deghost"] < v["residual_median"]


def test_ghost_mask_accepts_nested_lists():
    rng = np.random.default_rng(0)
    g = (0.4 + 0.08 * rng.random((18, 26))).tolist()
    f0 = [row[:] for row in g]
    for y in range(5, 12):
        for x in range(5, 12):
            f0[y][x] = 0.97
    rep = mod.ghost_mask([f0, g, [row[:] for row in g]])
    assert rep["ok"] and rep["ghost_ratio"] > 0.0


# ---------------------------------------------------------------------------
# 3. 曝光按实拍 EV（快门 + ISO + 光圈）与暗角校正
# ---------------------------------------------------------------------------

def test_iso_is_accounted_in_exposure_stops():
    s100 = {"exposure_time": 1 / 100, "iso": 100, "aperture": 2.8}
    s400 = {"exposure_time": 1 / 100, "iso": 400, "aperture": 2.8}
    assert mod.exposure_stops(s400) - mod.exposure_stops(s100) == \
        pytest.approx(2.0, abs=1e-9)
    # EV100 与曝光量互为相反数，保证换算自洽。
    assert mod.shooting_ev100(s100) == pytest.approx(-mod.exposure_stops(s100))


def test_ev_vignette_correction_removes_step_and_beats_shutter_only():
    rep = mod.ev_vignette_correct(1)
    assert rep["ok"] and rep["iso_accounted"] is True
    assert rep["step_after"] < 0.25 * max(rep["step_before"], 1e-9)
    assert rep["step_after"] < rep["step_naive"]
    # 暗角校正：四角增益 > 1 且有限。
    assert rep["vignette_corner_gain"] > 1.0
    assert np.all(np.isfinite(rep["vignette_gain"]))
    # 三帧增益围绕参考帧为 [2, 1, 0.5]（-1/0/+1 stop）。
    assert rep["gains"] == pytest.approx([2.0, 1.0, 0.5], rel=1e-6)


def test_vignette_field_darkens_corners():
    v = mod.vignette_field((40, 60), amount=0.34)
    assert v[0, 0] < 0.8
    assert v.max() > 0.999          # 画面中心附近暗角场约为 1
    assert v[20, 30] > 0.999        # 靠近画面中心，暗角增益几乎为 1
    g = mod.vignette_gain_field((40, 60), amount=0.34)
    assert g[0, 0] == pytest.approx(1 / v[0, 0], rel=1e-6)


def test_ev_correction_on_explicit_stack():
    ds = mod.make_exposure_stack()
    rep = mod.ev_vignette_correct(ds["frames"], ds["shots"])
    assert rep["ok"]
    assert rep["step_after"] < 0.05


# ---------------------------------------------------------------------------
# 4. 解码屏障：全部完整后才合并，半解码帧不可读
# ---------------------------------------------------------------------------

def test_decode_barrier_reorders_to_request_order():
    data = [np.full((4, 4), i + 1.0) for i in range(3)]
    handles = [mod.DecodeHandle(i, data=data[i]) for i in range(3)]
    for idx in (2, 0, 1):           # 强制乱序完成
        handles[idx].finish()
    rep = mod.decode_barrier(handles)
    assert rep["ok"] and rep["all_complete"]
    assert [im[0, 0] for im in rep["images"]] == [1.0, 2.0, 3.0]


def test_half_decoded_handle_is_not_published():
    h = mod.DecodeHandle(0, data=np.zeros((3, 3)))
    assert not h.complete
    with pytest.raises(mod.DecodeIncomplete):
        h.get(timeout=0.05)
    rep = mod.decode_barrier([h], timeout=0.15)
    assert rep["ok"] is False and rep["failures"]


def test_failed_decode_fails_barrier_instead_of_merging_partial():
    h = mod.DecodeHandle(0, data=np.zeros((2, 2)), fail=True)
    h.run()
    rep = mod.decode_barrier([h])
    assert rep["ok"] is False
    assert rep["failures"] and "error" in rep["failures"][0]


def test_load_bracket_returns_all_complete_and_brightness_order():
    rep = mod.load_bracket(1)
    assert rep["ok"] and rep["all_complete"]
    assert len(rep["images"]) == 3
    assert sorted(rep["brightness_order"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 5. 回滚恢复“全部”缓存
# ---------------------------------------------------------------------------

def test_rollback_all_self_check_restores_three():
    rep = mod.rollback_all(1)
    assert rep["ok"] and rep["all_restored"]
    assert rep["count"] == 3 and len(rep["restored_keys"]) == 3


def test_rollback_restores_every_custom_cache():
    caches = {f"cam_{i}": np.full((5, 5), float(i)) for i in range(4)}
    original = {k: v.copy() for k, v in caches.items()}
    snap = mod.snapshot_all(caches)
    for k in caches:
        caches[k] = np.full((5, 5), 99.0)
    caches["new_tmp"] = np.ones((2, 2))
    rep = mod.rollback_all(caches, snap)
    assert rep["all_restored"] and rep["count"] == 4
    assert "new_tmp" not in caches
    for k, v in original.items():
        assert np.array_equal(caches[k], v)


def test_merge_failure_rolls_back_all_caches():
    frames = [np.full((6, 6), 0.35 + 0.1 * i) for i in range(3)]
    caches = {f"frame_{i}": np.full((6, 6), float(i)) for i in range(3)}
    original = {k: v.copy() for k, v in caches.items()}
    rep = mod.merge(frames, caches=caches, simulate_failure=True)
    assert rep["ok"] is False
    assert rep["rolled_back"] is True
    assert rep["rollback"]["all_restored"] is True
    for k, v in original.items():
        assert np.array_equal(caches[k], v)
    # 合并中途新增的键也应被撤销。
    assert "merge_state" not in caches


# ---------------------------------------------------------------------------
# 6. 输出色调映射
# ---------------------------------------------------------------------------

def test_tone_map_compresses_hdr_and_is_monotonic():
    hdr = np.tile(np.linspace(0, 8, 128), (8, 1))
    rep = mod.tone_map_out(hdr)
    assert rep["ok"] and rep["monotonic"]
    assert rep["input_max"] > 1.0
    assert rep["output_max"] <= 1.0 + 1e-6
    assert np.all(np.isfinite(rep["image"]))


def test_tone_map_bit_depths():
    hdr = np.linspace(0, 4, 64).reshape(8, 8)
    r8 = mod.tone_map_out(hdr, bit_depth=8)
    r16 = mod.tone_map_out(hdr, bit_depth=16)
    assert r8["out"].dtype == np.uint8 and r8["bit_depth"] == 8
    assert r16["out"].dtype == np.uint16 and r16["bit_depth"] == 16
    assert r16["unique_levels"] > r8["unique_levels"]


def test_tone_map_writes_16bit_png(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "out16.png"
    rep = mod.tone_map_out(np.linspace(0, 6, 256).reshape(16, 16),
                          bit_depth=16, path=str(path))
    assert rep["ok"] and rep["written"] and path.exists()
    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert back is not None and back.dtype == np.uint16


# ---------------------------------------------------------------------------
# 7. 端到端 compose / 审计 / 指标 / fallback / 确定性
# ---------------------------------------------------------------------------

def test_compose_end_to_end():
    rep = mod.compose(1)
    assert rep["ok"], rep.get("error")
    assert rep["output"] is not None
    assert rep["image"].shape[-1] == 3      # demo 是 RGB
    assert rep["load"]["all_complete"] is True
    assert rep["fixes"]["decode_barrier_all_complete"] is True
    assert rep["fixes"]["rollback_all_caches"] is True


def test_audit_one_reports_all_six_fixes():
    a = mod.audit(1)
    assert a["ok"], a["issues"]
    assert a["issue_count"] == 0 and a["issues"] == []
    for key in ("local_reference_alignment", "ghost_brightness_diff_mask",
                "real_ev_iso_vignette", "decode_barrier_all_complete",
                "rollback_all_caches", "tone_map_output"):
        assert a["fixes"][key] is True
        assert a[key] is True


def test_determinism_bit_identical():
    rep = mod.determinism(1, repeats=3)
    assert rep["ok"]
    assert len(rep["digests"]) == 3
    assert len(set(rep["digests"])) == 1


def test_fallback_is_safe_and_non_none():
    fb = mod.fallback("merge_failed")
    assert fb["ok"] and fb["strategy"] == "reference_frame"
    assert fb["image"] is not None
    assert fb["params"]["max_boost"] <= 1.5


def test_metrics_two_images_psnr():
    a = np.zeros((4, 4)); b = np.ones((4, 4)) * 0.5
    r = mod.metrics(a, b)
    assert r["mse"] == pytest.approx(0.25)
    assert r["psnr"] > 0 and r["mae"] == pytest.approx(0.5)


def test_sort_by_brightness():
    frames = [np.full((2, 2), 0.7), np.full((2, 2), 0.1), np.full((2, 2), 0.4)]
    assert mod.sort_by_brightness(frames) == [1, 2, 0]


def test_load_and_compose_from_image_paths(tmp_path):
    cv2 = pytest.importorskip("cv2")
    base = mod.make_scene((36, 48))
    paths = []
    for i, gain in enumerate((0.5, 1.0, 1.5)):
        p = tmp_path / f"b{i}.png"
        cv2.imwrite(str(p), np.clip(base * gain * 255, 0, 255).astype(np.uint8))
        paths.append(str(p))
    loaded = mod.load_bracket(paths)
    assert loaded["ok"] and loaded["all_complete"] and len(loaded["images"]) == 3
    out = tmp_path / "final.png"
    comp = mod.compose(paths, output_path=str(out))
    assert comp["ok"], comp.get("error")
    assert out.exists() and out.stat().st_size > 0
