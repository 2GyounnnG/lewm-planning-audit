# Reacher joint-angle probe audit

The sealed R5 H1b Reacher probe truth uses the direct `qpos` components indexed by the
formal protocol as `joint_angle_0` and `joint_angle_1` (see `FIT_PROTOCOL.json`,
`reacher.position.indices = [0,1]`).  The 100 saved EVAL anchors are repeated across
the three MLP seeds and the single/three-frame histories, so truth values are identical
across those conditions.

From `r5_execution/h1b/reports/main/H1B_ALL_RAW_VALUES.csv`, filtered to
`task=reacher`, `target=state`, and the two `joint_angle_*` components:

| component | truth range (rad) | values with `abs(angle) > pi` | fraction |
|---|---:|---:|---:|
| `joint_angle_0` (first arm/shoulder joint) | [-4.9119858411, 5.3129646879] | 78 / 600 repeated rows (13 / 100 anchors) | 0.130 |
| `joint_angle_1` (second arm/wrist/elbow joint) | [-2.8786613580, 2.7981688139] | 0 / 600 | 0.000 |

Thus the first joint angle is an unwrapped hinge coordinate retained in cumulative
MuJoCo `qpos` radians; it is not reduced modulo `2 pi`.  The second joint happens to
remain inside `[-pi, pi]` for these anchors.  This is a coordinate/trajectory fact,
not evidence that the first angle is intrinsically harder to predict.

For the single-frame three-seed MLP mean, the frozen information table reports
`R^2(joint_angle_0) = 0.168342` (95% conditional interval [0.026126, 0.322244]) and
`R^2(joint_angle_1) = 0.999542` ([0.999402, 0.999638]).  These are the values to
report as approximately 0.17 and 1.00, respectively, without a blanket claim that
joint angles are uniformly easy to decode.

Source hashes:

* `r5_execution/h1b/reports/main/H1B_ALL_RAW_VALUES.csv`: `3c942cda6c26d5bb67543fdfc95cff3641f529039e97475bdf0c91f545ef61fe`.
* `r5_execution/h1b/reports/main/SINGLE_FRAME_INFORMATION_TABLE.csv`: `d78137939cf6f37decf6b39a85a63567c80c4d82392da850258eafc03379122b`.
* Local evidence copy of the information table: `paper_w1/evidence/r5/SINGLE_FRAME_INFORMATION_TABLE.csv`, SHA-256 `f757db0f2511736854984f64e53be9494fd3458f5570662cdefeb9c825283f89`.
* Reacher probe archive: `r5_execution/h1b/archives/reacher_probe_recovery_v1.tar.gz`, SHA-256 `d6323b158adbe3ce52c1662a386f8a5ed066cdceae659562bd9e924363982bfa`.
* Formal protocol (`qpos` mapping): `r5_execution/h1b/FIT_PROTOCOL.json`, SHA-256 `9c7099361adc44a9df5a88b88d2b6bffd6e44c9cb3fb666f4eb31de76203a004`.

