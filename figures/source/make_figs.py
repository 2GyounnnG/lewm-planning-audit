"""Redesigns of Figures 2--5, read directly from archived local evidence tables.

All values are loaded from the CSV/JSON mirrors below; no values are hand
transcribed and the script performs no training, inference, or closed-loop run.
Sources (local copies):
  FIG2  paper_w1/evidence/r5/FOUR_TASK_MAIN_TABLE_V2.csv (OPEN_LOOP and CLOSED_LOOP rows)
  FIG3  paper_w1/evidence/r5/H3X_CONTEXT_MATCHED_REFIT_EXPLORATORY_TABLE.csv (LATENT_MSE_FP32)
        paper_w1/evidence/r5/SINGLE_FRAME_INFORMATION_TABLE.csv (MLP_3SEED_MEAN_NOT_ENSEMBLE, R2)
  FIG4  paper_w1/evidence/r5/REAL_HISTORY_REPLANNING_TABLE.csv; H3X table (SUCCESS_RATE);
        paper_w1/evidence/r6/R6_RAW_VALUES.csv (fresh-case endpoint)
  FIG5  paper_w1/evidence/r4/R4_ATTRIBUTION_TABLE.csv plus R8 M4 100-case oracle tables
"""
from pathlib import Path
import pandas as pd
import json
import hashlib
from decimal import Decimal, ROUND_HALF_UP
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path(__file__).parent
FONT_FAMILY = "DejaVu Sans"
font_candidates = [
    ("/System/Library/Fonts/Helvetica.ttc", "Helvetica"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "Arial"),
    ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "Liberation Sans"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "Liberation Sans"),
]
for font_path, family in font_candidates:
    if Path(font_path).exists():
        try:
            font_manager.fontManager.addfont(font_path)
            for face in ("-Bold", "-Italic", "-BoldItalic"):
                extra = font_path.replace("-Regular", face)
                if extra != font_path and Path(extra).exists():
                    font_manager.fontManager.addfont(extra)
            FONT_FAMILY = family
            break
        except Exception:
            pass

# ---------------------------------------------------------------- style
INK, INK2, MUTED, HAIR, GRID = "#0B0B0B", "#52514E", "#898781", "#C3C2B7", "#E1E0D9"
OFFICIAL, REFIT, HIST, CTX = "#6E6D69", "#2A78D6", "#EB6834", "#1BAF7A"
W = 5.40  # inches: elsarticle preprint text width; final layout should place these full width

mpl.rcParams.update({
    "font.family": FONT_FAMILY, "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": HAIR, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "mathtext.fontset": "custom", "mathtext.rm": FONT_FAMILY, "mathtext.it": FONT_FAMILY + ":italic",
})


def panel(ax, letter, title, dx=0.0, dy=7.0):
    """Bold panel letter + title above the axes; dx/dy in points from the axes' top-left corner."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color=INK, ha="left", va="bottom")
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(dx + 9.5, dy), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="left", va="bottom")


def panel_at(ax, letter, title, xfig, dy=7.0):
    """Panel letter + title at a fixed figure-fraction x, just above the axes."""
    ax.annotate(letter, xy=(xfig, 1), xycoords=("figure fraction", "axes fraction"), xytext=(0, dy),
                textcoords="offset points", fontsize=8.5, fontweight="bold", color=INK, ha="left", va="bottom")
    if title:
        ax.annotate(title, xy=(xfig, 1), xycoords=("figure fraction", "axes fraction"), xytext=(9.5, dy),
                    textcoords="offset points", fontsize=7.5, color=INK, ha="left", va="bottom")


def m(x, fmt="{:.1f}", sign=False):
    """Format with a true minus sign."""
    # Use decimal half-up rounding for reported one-decimal percentage points
    # (e.g., 13.25 -> 13.3), rather than Python's binary tie-to-even display.
    try:
        precision = int(fmt.split(".")[1].split("f")[0])
        q = Decimal("1").scaleb(-precision)
        xx = Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP)
        body = f"{xx:.{precision}f}"
    except Exception:
        body = fmt.format(x)
    t = (("+" if x > 0 else "") if sign else "") + body
    return t.replace("-", "\u2212")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- data (read directly from the archived CSV/JSON tables)
ROOT = Path(__file__).resolve().parents[2]
R5 = ROOT / "evidence" / "r5"
R4 = ROOT / "evidence" / "r4"
R6 = ROOT / "evidence" / "r6"
FT_PATH = R5 / "FOUR_TASK_MAIN_TABLE_V2.csv"
H3_PATH = R5 / "H3X_CONTEXT_MATCHED_REFIT_EXPLORATORY_TABLE.csv"
PROBE_PATH = R5 / "SINGLE_FRAME_INFORMATION_TABLE.csv"
RH_PATH = R5 / "REAL_HISTORY_REPLANNING_TABLE.csv"
ATTR_PATH = R4 / "R4_ATTRIBUTION_TABLE.csv"
R6_MAIN_PATH = R6 / "R6_MAIN_TABLE.csv"
R6_RAW_PATH = R6 / "R6_RAW_VALUES.csv"
R6_STATS_PATH = R6 / "R6_STATS.json"
R7 = ROOT / "evidence" / "r7"
R8 = ROOT / "evidence" / "r8"
R8_MAIN_PATH = R8 / "reports" / "R8_MAIN_TABLE_V2.csv"
R8_SECONDARY_PATH = R8 / "reports" / "R8_SECONDARY_TABLE_V2.csv"
R8_E2_FIX_PATH = R8 / "fix" / "f1" / "R8_M1_E2_CORRECTED_TABLE.csv"
M4_MAIN_PATH = R8 / "m4" / "M4_MAIN_TABLE.csv"
M4_SECONDARY_PATH = R8 / "m4" / "M4_SECONDARY_TABLE.csv"
M4_RAW_PATH = R8 / "m4" / "M4_ALL_RAW_VALUES.csv"
main_df = pd.read_csv(FT_PATH)
h3x_df = pd.read_csv(H3_PATH)
info_df = pd.read_csv(PROBE_PATH)
hist_df = pd.read_csv(RH_PATH)
r4_attr_df = pd.read_csv(ATTR_PATH)
r6_main_df = pd.read_csv(R6_MAIN_PATH)
r6raw_df = pd.read_csv(R6_RAW_PATH)
r6_stats = json.loads(R6_STATS_PATH.read_text())
r8_main_df = pd.read_csv(R8_MAIN_PATH)
r8_secondary_df = pd.read_csv(R8_SECONDARY_PATH)
r8_e2_fix_df = pd.read_csv(R8_E2_FIX_PATH)
m4_main_df = pd.read_csv(M4_MAIN_PATH)
m4_secondary_df = pd.read_csv(M4_SECONDARY_PATH)
m4_raw_df = pd.read_csv(M4_RAW_PATH)
r7_a_df = pd.read_csv(R7 / "reports" / "A_MAIN_TABLE.csv")
W6_PATH = ROOT / "evidence" / "w6" / "W6_POSTHOC_TABLE.csv"
w6_df = pd.read_csv(W6_PATH)
R10_PATH = ROOT / "evidence" / "r10" / "R10_MAIN_TABLE.csv"
r10_df = pd.read_csv(R10_PATH)
def r10(endpoint):
    rr = r10_df[r10_df.endpoint == endpoint].iloc[0]
    return float(rr.point_pp), float(rr.ci95_low_pp), float(rr.ci95_high_pp)

# Fresh-case endpoint: case means across all model/stream pairs, followed by
# the preregistered case bootstrap.  This re-expresses the sealed calculation
# from R6_RAW_VALUES.csv; no trajectories are generated here.
_p = r6raw_df[r6raw_df.history == "H_POLICY"].groupby("case_id").success.mean()
_r = r6raw_df[r6raw_df.history == "H_REAL3_REPLAN"].groupby("case_id").success.mean()
_d = (_r - _p).dropna().to_numpy()
_pe = r6_stats["primary_endpoint"]
_rng = np.random.default_rng(int(_pe["bootstrap_seed_uint64"]))
_idx = _rng.integers(0, len(_d), size=(int(_pe["bootstrap"]), len(_d)))
_boot = _d[_idx].mean(axis=1)
R6_ENDPOINT = (100 * float(_d.mean()), 100 * float(np.quantile(_boot, 0.025)), 100 * float(np.quantile(_boot, 0.975)))

TASKS = ["PushT", "Reacher", "TwoRoom", "Cube"]
TASK_KEY = {"PushT": "pusht", "Reacher": "reacher", "TwoRoom": "tworoom", "Cube": "cube"}
H = [1, 2, 5]
REFIT_ARM = "FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE"
RH_REFIT_ARM = "FIXED3_REFIT_MEAN_NOT_ENSEMBLE"

def row(df, **kwargs):
    q = df
    for k, v in kwargs.items():
        if v is None: q = q[q[k].isna()]
        else: q = q[q[k] == v]
    if len(q) != 1:
        raise ValueError(f"Expected one row for {kwargs}, got {len(q)}")
    return q.iloc[0]

OFFLINE, CLOSED = {}, {}
for label in TASKS:
    task = TASK_KEY[label]
    vals = []
    for h in H:
        off = row(main_df, task=task, evaluation_kind="OPEN_LOOP", arm="H0", horizon_macro=h)["latent_raw_MSE"]
        ref = row(main_df, task=task, evaluation_kind="OPEN_LOOP", arm=REFIT_ARM, horizon_macro=h)["latent_raw_MSE"]
        vals.append(100.0 * (ref - off) / off)
    OFFLINE[label] = vals
    # Cube carries an explicit R3 stream; the other main-table tasks have a
    # blank stream because their protocol has a single fixed stream.
    stream = "R3_ORIGINAL" if task == "cube" else None
    ro = row(main_df, task=task, evaluation_kind="CLOSED_LOOP_CEM", arm="H0", stream=stream, horizon_macro=0)
    rr = row(main_df, task=task, evaluation_kind="CLOSED_LOOP_CEM", arm=REFIT_ARM, stream=stream, horizon_macro=0)
    CLOSED[label] = (float(ro.success_percent), float(rr.success_percent), float(rr.delta_success_pp),
                     float(rr.delta_case_ci95_low), float(rr.delta_case_ci95_high))

# Reacher MSE bands: official, refit mean, and context-matched refit.
MSE = {}
for qname, hist, arm in [("three", "H_REAL3", "H0"), ("three", "H_REAL3", "FIXED3_R3_MEAN_NOT_ENSEMBLE"),
                          ("context_three", "H_REAL3", "FIXED3_H3X_MEAN_NOT_ENSEMBLE"),
                          ("single", "H_POLICY", "H0"), ("single", "H_POLICY", "FIXED3_R3_MEAN_NOT_ENSEMBLE"),
                          ("context_single", "H_POLICY", "FIXED3_H3X_MEAN_NOT_ENSEMBLE")]:
    key = {"H0":"official", "FIXED3_R3_MEAN_NOT_ENSEMBLE":"refit", "FIXED3_H3X_MEAN_NOT_ENSEMBLE":"context"}[arm]
    if qname.startswith("context_"): key = "context"
    out = []
    for h in H:
        rr = row(h3x_df, metric="LATENT_MSE_FP32", arm=arm, history_kind=hist, horizon_macro=float(h))
        out.append((float(rr["mean"]), float(rr["conditional95_low"]), float(rr["conditional95_high"])))
    MSE[(key, "three" if hist == "H_REAL3" else "single")] = out

VEL_R2 = []
for task, comp, label in [
    ("reacher", "joint_velocity_0", "Reacher shoulder velocity"),
    ("reacher", "joint_velocity_1", "Reacher wrist velocity"),
    ("pusht", "agent_vx", "PushT agent $v_x$"),
    ("pusht", "agent_vy", "PushT agent $v_y$"),
]:
    s1 = row(info_df, task=task, model="MLP_3SEED_MEAN_NOT_ENSEMBLE", history="single", metric="R2", component=comp)["estimate"]
    s3 = row(info_df, task=task, model="MLP_3SEED_MEAN_NOT_ENSEMBLE", history="three", metric="R2", component=comp)["estimate"]
    VEL_R2.append((label, float(s1), float(s3)))

# Forest data are all read from the corresponding archived summary tables.
FOREST = []
def add_hist(label, arm, color, filled=True):
    rr = row(hist_df, task="reacher", metric="success_difference_REAL3_minus_POLICY", arm=arm, stream="MEAN_OF_THREE_STREAMS_WITHIN_CASE")
    FOREST.append(("Change the history (real − single)", label, 100*float(rr.estimate), 100*float(rr.conditional95_low), 100*float(rr.conditional95_high), color, filled))
# Predictor intervention rows (single-frame query) come first in the forest.
rr = row(h3x_df, metric="SUCCESS_RATE", arm="FIXED3_R3_MEAN_NOT_ENSEMBLE", history_kind="OFFICIAL_DEFAULT", horizon_macro=None)
FOREST.append(("Change the predictor", "Refit − official, single-frame query", 100*float(rr.difference_vs_H0), 100*float(rr.difference_vs_H0_low), 100*float(rr.difference_vs_H0_high), REFIT, True))
rr = row(h3x_df, metric="SUCCESS_RATE", arm="FIXED3_H3X_MEAN_NOT_ENSEMBLE", history_kind="OFFICIAL_DEFAULT", horizon_macro=None)
# The exploratory context-matched comparison is recorded as a difference in its own table.
FOREST.append(("Change the predictor", "Context-matched − refit, single-frame", 100*float(rr.difference_vs_R3_FIXED3), 100*float(rr.difference_vs_R3_FIXED3_low), 100*float(rr.difference_vs_R3_FIXED3_high), CTX, False))
# History intervention rows, then the task control and the fresh-case endpoint.
rr = row(hist_df, task="reacher", metric="success_difference_vs_H0", arm=RH_REFIT_ARM, stream="MEAN_OF_THREE_STREAMS_WITHIN_CASE", history_kind="H_REAL3_REPLAN")
FOREST.append(("Change the predictor", "Refit − official, real-history replanning", 100*float(rr.estimate), 100*float(rr.conditional95_low), 100*float(rr.conditional95_high), REFIT, False))
add_hist("Official predictor", "H0", HIST)
add_hist("Refit predictor", RH_REFIT_ARM, HIST)
FOREST.append(("Change the history (real − single)", "Fresh cases, preregistered", *R6_ENDPOINT, HIST, True))
# R10 module A: the planning library's own history option (official predictor, history set).
FOREST.append(("Change the history (real − single)", "Library option, history_len 3 − 1", *r10("E_A1"), HIST, False))
rr = row(hist_df, task="pusht", metric="success_difference_REAL3_minus_POLICY", arm=RH_REFIT_ARM, stream="MEAN_OF_THREE_STREAMS_WITHIN_CASE")
FOREST.append(("Control tasks", "PushT: real − single, refit", 100*float(rr.estimate), 100*float(rr.conditional95_low), 100*float(rr.conditional95_high), OFFICIAL, True))
# R10 modules B and C: TwoRoom and Cube real-history replanning on the fresh refit sets.
FOREST.append(("Control tasks", "TwoRoom: real − single, four predictors", *r10("E_B"), OFFICIAL, True))
FOREST.append(("Control tasks", "Cube: real − single, four predictors", *r10("E_C"), OFFICIAL, True))

STREAMS = {}
for arm, name, marker in [("H0", "Official", "o"), (RH_REFIT_ARM, "Refit (mean)", "s")]:
    singles, reals = [], []
    for stream in ["R3_ORIGINAL", "R4_ALT_CEM_1", "R4_ALT_CEM_2"]:
        a = row(hist_df, task="reacher", metric="success_fraction", arm=arm, stream=stream, history_kind="H_POLICY")
        b = row(hist_df, task="reacher", metric="success_fraction", arm=arm, stream=stream, history_kind="H_REAL3_REPLAN")
        singles.append(100*float(a.estimate)); reals.append(100*float(b.estimate))
    STREAMS[name] = (singles, reals)

# PushT fixed-menu metrics and simulator reference.
def attr(metric, arm):
    return row(r4_attr_df, scope="INITIAL_FIXED64", task="pusht", metric=metric, arm=arm,
               history_kind=None, horizon_raw=None, task_time_reduction="ANYTIME")
dt0 = attr("D_total", "H0"); dt1 = attr("D_total", RH_REFIT_ARM)
sp0 = attr("spearman_model_sim_lat", "H0"); sp1 = attr("spearman_model_sim_lat", RH_REFIT_ARM)
s0 = row(main_df, task="pusht", evaluation_kind="CLOSED_LOOP_CEM", arm="H0", stream=None, horizon_macro=0)
s1 = row(main_df, task="pusht", evaluation_kind="CLOSED_LOOP_CEM", arm=REFIT_ARM, stream=None, horizon_macro=0)
# The R8 M4 oracle extension supersedes the 20-case panel; the original cases are included.
orc = m4_main_df[m4_main_df.stream == "POOLED_CASE_MEAN_3_STREAMS"].iloc[0]
# Success rates: case means over the three planner streams, then the mean over 100 cases.
M4_RATE = (100 * m4_raw_df.groupby(["arm", "case_id"]).success.mean().groupby("arm").mean()).to_dict()
_m4p = m4_secondary_df[m4_secondary_df.stream == "POOLED_CASE_MEAN_3_STREAMS"]
M4_DELTA = {r.endpoint.split(" - ")[0]: (100 * float(r.estimate), 100 * float(r.ci_low), 100 * float(r.ci_high))
            for r in _m4p.itertuples()}
PUSHT = [
    ("Selection loss", "lower is better", float(dt0.estimate), float(dt1.estimate), float(dt1.difference_vs_reference), float(dt1.difference95_low), float(dt1.difference95_high), "{:.3f}", ("official", "refit")),
    ("Rank correlation", "higher is better", float(sp0.estimate), float(sp1.estimate), float(sp1.difference_vs_reference), float(sp1.difference95_low), float(sp1.difference95_high), "{:.3f}", ("official", "refit")),
    ("Closed-loop success (%)", "100 cases, stream A", float(s0.success_percent), float(s1.success_percent), float(s1.delta_success_pp), float(s1.delta_case_ci95_low), float(s1.delta_case_ci95_high), "{:.0f}", ("official", "refit")),
    ("Scoring success (%)", "100 cases, 3 streams", M4_RATE["H0_MENU_RERANK"], M4_RATE["SIM_LAT_RERANK"], M4_RATE["SIM_TASK_RERANK"], 0.0, 0.0, "{:.1f}", ("latent", "sim-lat", "sim-task"))]

# R8-expanded forest for the predictor contrast panel. All rows are read from
# the archived R5/R7/R8 summary tables; invalid R8 E2 and PRED_PAST rows are omitted.
FOREST2 = []
for label in TASKS:
    off, ref, d, lo, hi = CLOSED[label]
    FOREST2.append((label + " · original", d, lo, hi, REFIT))
for task, endpoint, label in [("reacher", "fixed_three_refit_mean_minus_H0", "Reacher · fresh"),
                              ("pusht", "fixed_three_refit_mean_minus_H0", "PushT · fresh"),
                              ("tworoom", "fixed_three_refit_mean_minus_H0", "TwoRoom · fresh"),
                              ("cube", "fixed_three_refit_mean_minus_H0", "Cube · fresh")]:
    rr = row(r8_main_df, module="M2", task=task, endpoint=endpoint)
    FOREST2.append((label, 100*float(rr.point), 100*float(rr.ci95_low), 100*float(rr.ci95_high), REFIT))
# W6 post hoc pool of the three Reacher case sets (single-frame query, three
# streams), read from the W6 table produced by scripts/w6_posthoc.py.
_pool = w6_df[w6_df["sample"].str.startswith("reacher_pooled") & (w6_df["quantity"] == "contrast_0")].iloc[0]
FOREST2.append(("Reacher · 3 sets pooled", float(_pool.estimate_pp), float(_pool.ci95_low_pp), float(_pool.ci95_high_pp), REFIT))
for task, label in [("reacher", "Reacher · stress"), ("pusht", "PushT · stress")]:
    rr = r7_a_df[(r7_a_df.endpoint == "P2") & (r7_a_df.task == task)].iloc[0]
    FOREST2.append((label, 100*float(rr.point), 100*float(rr.ci95_low), 100*float(rr.ci95_high), REFIT))
for unit, label in [("U1", "Reacher · budget ×2"), ("U2", "Reacher · offset ×2")]:
    rr = row(r8_main_df, module="M1", unit=unit, endpoint="E3_refit_minus_H0")
    FOREST2.append((label, 100*float(rr.point), 100*float(rr.ci95_low), 100*float(rr.ci95_high), REFIT))
for task, label in [("pusht", "PushT · search ×3"), ("reacher", "Reacher · search ×3")]:
    rr = row(r8_main_df, module="M6", task=task, endpoint="mean_REFIT_minus_H0_at_CEM900")
    FOREST2.append((label, 100*float(rr.point), 100*float(rr.ci95_low), 100*float(rr.ci95_high), REFIT))

# ---------------------------------------------------------------- Figure 2
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.25), gridspec_kw={"width_ratios": [0.85, 1.35], "wspace": 0.62})
    y = np.arange(len(TASKS))[::-1]
    shapes = {1: ("o", REFIT, 24), 2: ("o", "white", 20), 5: ("D", "white", 15)}
    for yi, t in zip(y, TASKS):
        v = OFFLINE[t]
        a.plot([min(v), max(v)], [yi, yi], color=HAIR, lw=1.0, zorder=1)
        for h, val in zip(H, v):
            mk, fc, sz = shapes[h]
            a.scatter(val, yi, s=sz, marker=mk, facecolor=fc, edgecolor=REFIT, linewidth=1.0, zorder=3)
        if min(v) > -70:
            a.text(min(v) - 4, yi, m(v[0], "{:.0f}") + "%", ha="right", va="center", fontsize=6.8, color=INK2)
        else:
            a.text(max(v) + 4, yi, m(v[0], "{:.0f}") + "%", ha="left", va="center", fontsize=6.8, color=INK2)
    a.set_xlim(-105, 3); a.set_xticks([-100, -75, -50, -25, 0])
    a.set_xticklabels([m(t, "{:.0f}") for t in [-100, -75, -50, -25, 0]])
    a.axvline(0, color=HAIR, lw=0.6)
    a.set_yticks(y); a.set_yticklabels(TASKS); a.tick_params(axis="y", length=0, labelcolor=INK)
    a.spines["left"].set_visible(False); a.set_ylim(-0.6, 3.6)
    a.set_xlabel("Change in offline latent error (%)")
    a.xaxis.grid(True); a.set_axisbelow(True)
    hd = [plt.Line2D([], [], marker=shapes[h][0], ls="", markerfacecolor=shapes[h][1], markeredgecolor=REFIT,
                     markersize=4.2 if h != 5 else 3.6, label=f"h = {h}") for h in H]
    a.legend(handles=hd, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=1, handletextpad=0.2, labelspacing=0.25)
    panel(a, "a", "Offline error (three-frame query)", dx=-42)

    # Expanded predictor-contrast forest from all valid conditions.
    yy = np.arange(len(FOREST2))[::-1]
    for yi, (lab, d, lo, hi, col) in zip(yy, FOREST2):
        filled = ("original" in lab or "fresh" in lab)
        pooled = "pooled" in lab
        b.plot([lo, hi], [yi, yi], color=col, lw=1.1, solid_capstyle="round")
        b.scatter(d, yi, s=16 if pooled else 18, marker="D" if pooled else "o",
                  facecolor=col if filled else "white", edgecolor=col, linewidth=0.9, zorder=3)
    b.axvline(0, color=INK2, lw=0.7)
    # The fresh-case history contrast is a reference band; it is not a
    # predictor effect and is shown only to keep the scale interpretable.
    b.axvspan(R6_ENDPOINT[1], R6_ENDPOINT[2], color="#F3B47A", alpha=0.18, zorder=0)
    b.text(0.5 * (R6_ENDPOINT[1] + R6_ENDPOINT[2]), (len(FOREST2) - 1) / 2, "Reacher history gain, fresh cases",
           rotation=90, ha="center", va="center", fontsize=6.3, color="#B7602D")
    b.set_xlim(-12, 18); b.set_xticks([-10, -5, 0, 5, 10, 15])
    b.set_xticklabels([m(t, "{:.0f}", sign=True) if t else "0" for t in [-10, -5, 0, 5, 10, 15]])
    b.set_yticks(yy); b.set_yticklabels([x[0] for x in FOREST2], fontsize=6.5); b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False); b.set_ylim(-1, len(FOREST2))
    b.set_xlabel("Refit − official success (pp, 95% CI)")
    b.xaxis.grid(True); b.set_axisbelow(True)
    panel(b, "b", "Refit − official success")
    save(fig, "fig2_same_plans")


# ---------------------------------------------------------------- Figure 3
def fig3():
    fig = plt.figure(figsize=(W, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.40)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[0, 2])
    styles = {"official": (OFFICIAL, "official"), "refit": (REFIT, "refit"), "context": (CTX, "context-\nmatched")}
    for ax, q, ttl, letter in [(a, "three", "Three-frame query", "a"), (b, "single", "Single-frame query", "b")]:
        for k, (col, lab) in styles.items():
            mm = np.array(MSE[(k, q)])
            ax.fill_between(H, mm[:, 1], mm[:, 2], color=col, alpha=0.13, lw=0)
            ax.plot(H, mm[:, 0], color=col, lw=1.6, marker="o", ms=3.4, mec="white", mew=0.6)
        ax.set_yscale("log"); ax.set_ylim(5e-4, 2.5e-1); ax.set_xticks(H); ax.set_xlim(0.7, 5.3)
        ax.set_xlabel("Horizon (macro steps)")
        ax.yaxis.grid(True, which="major"); ax.set_axisbelow(True)
        panel(ax, letter, ttl)
    a.set_ylabel("Reacher latent MSE")
    b.tick_params(axis="y", labelleft=False)
    nudge = {"official": 4, "refit": 0, "context": -5}
    for k, (col, lab) in styles.items():
        a.annotate(lab, xy=(5, MSE[(k, "three")][2][0]), xytext=(4, nudge[k]), textcoords="offset points",
                   color=INK2, fontsize=6.5, ha="left", va="center", linespacing=0.95)
    y_hi, y_lo = MSE[("refit", "single")][0][0], MSE[("refit", "three")][0][0]
    b.annotate("", xy=(1, y_hi * 0.88), xytext=(1, y_lo * 1.15), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7))
    b.plot([1], [y_lo], marker="o", ms=3.4, mfc="white", mec=REFIT, mew=0.9)
    b.annotate("×70 for the refit:\nsingle- vs. three-frame\nerror (open circle)", xy=(1, 0.008), xytext=(5, 0),
               textcoords="offset points", fontsize=6.5, color=INK2, va="center")
    b.annotate("all three predictors\nnearly coincide", xy=(5, 0.13), xytext=(0, 4), textcoords="offset points",
               fontsize=6.5, color=INK2, ha="right", va="bottom")

    rows = VEL_R2
    yy = np.arange(len(rows))[::-1]
    for yi, (lab, s1, s3) in zip(yy, rows):
        c.plot([s1, s3], [yi, yi], color=HAIR, lw=1.2, zorder=1)
        c.scatter(s1, yi, s=22, facecolor="white", edgecolor=OFFICIAL, linewidth=1.0, zorder=3)
        c.scatter(s3, yi, s=22, color=INK2, zorder=3)
    c.set_yticks(yy); c.set_yticklabels([r[0].replace("Reacher ", "Reacher\n").replace("PushT ", "PushT\n") for r in rows])
    c.tick_params(axis="y", length=0, labelcolor=INK, labelsize=6.6)
    c.yaxis.tick_right(); c.spines["left"].set_visible(False)
    c.set_xlim(-0.12, 0.8); c.set_xticks([0, 0.25, 0.5, 0.75]); c.set_xticklabels(["0", "0.25", "0.50", "0.75"])
    c.axvline(0, color=HAIR, lw=0.6); c.set_ylim(-0.6, 3.9)
    c.set_xlabel("Velocity probe $R^2$")
    c.xaxis.grid(True); c.set_axisbelow(True)
    hd = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=OFFICIAL, ms=4.0, label="1 frame"),
          plt.Line2D([], [], marker="o", ls="", color=INK2, ms=4.0, label="3 frames")]
    c.legend(handles=hd, loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=2, handletextpad=0.2, columnspacing=0.8)
    panel(c, "c", "Velocity from frames")
    save(fig, "fig3_query_regime")


# Valid R8 history rows and operating-point decomposition for Figure 4.
for label, point, low, high, col, filled in [
    ("Three real frames − static context", "H_REAL3_minus_REPEAT3", None, None, HIST, True),
    ("Two real frames − single frame", "REAL2_minus_H_POLICY", None, None, HIST, False),
    ("Static context − single frame", "REPEAT3_minus_H_POLICY", None, None, HIST, False),
    ("Full history − single frame", "FULL_HISTORY - H_POLICY", None, None, HIST, False),
]:
    if point in {"H_REAL3_minus_REPEAT3", "REAL2_minus_H_POLICY", "REPEAT3_minus_H_POLICY"}:
        rr = row(r8_main_df if point == "H_REAL3_minus_REPEAT3" else r8_secondary_df, module="M3", endpoint=point)
    else:
        # M5 rows are in the V2 main table and labelled descriptive.
        rr = row(r8_main_df, module="M5", task="reacher", endpoint=point)
    FOREST.append(("Mechanism / protocol", label, 100*float(rr.point), 100*float(rr.ci95_low), 100*float(rr.ci95_high), col, filled))
# R10 module D: the two components of real history, unpaired (preregistered).
FOREST.append(("Mechanism / protocol", "Real frames, null actions − single frame", *r10("E_D1"), HIST, False))
FOREST.append(("Mechanism / protocol", "Current-frame copies, real actions − single frame", *r10("E_D2"), HIST, False))
# Operating-point E1 rows are read from the sealed summaries above.  The two
# corrected E2 rows are read directly from the R8-FIX table and are plotted as
# open markers so the estimate is visibly distinct from E1.
BUDGET_POINTS = [
    ("50 / 25", *R6_ENDPOINT, "e1"),
]
for _unit, _label in [("U2", "50 / 50"), ("U1", "100 / 25")]:
    _rr = row(r8_main_df, module="M1", unit=_unit, endpoint="E1_H_REAL3_minus_H_POLICY")
    BUDGET_POINTS.append((_label, 100*float(_rr.point), 100*float(_rr.ci95_low),
                          100*float(_rr.ci95_high), "e1"))
_rr = r7_a_df[(r7_a_df.endpoint == "P1") & (r7_a_df.task == "reacher")].iloc[0]
BUDGET_POINTS.append(("100 / 50", 100*float(_rr.point), 100*float(_rr.ci95_low),
                      100*float(_rr.ci95_high), "e1"))
_e2 = r8_e2_fix_df[r8_e2_fix_df.endpoint == "E2_budget50_history_effect_minus_budget100_history_effect"]
for _, _rr in _e2.iterrows():
    _offset = "25" if str(_rr.unit).startswith("offset25") else "50"
    BUDGET_POINTS.append((f"E2 {_offset}", 100*float(_rr.point),
                          100*float(_rr.ci95_low), 100*float(_rr.ci95_high), "e2"))

# ---------------------------------------------------------------- Figure 4
def fig4():
    fig = plt.figure(figsize=(W, 5.3))
    gs_top = fig.add_gridspec(1, 1, left=0.425, right=0.90, top=0.965, bottom=0.46)
    gs_bot = fig.add_gridspec(1, 2, left=0.095, right=0.985, top=0.31, bottom=0.06,
                              width_ratios=[1.0, 1.3], wspace=0.30)
    a = fig.add_subplot(gs_top[0, 0])
    b, c = fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])

    labels, ypos, y, last = [], [], 0.0, None
    for g, lab, est, lo, hi, col, filled in FOREST:
        if g != last:
            if last is not None:
                y -= 0.45
            a.annotate(g, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-4, 0), textcoords="offset points",
                       ha="right", va="center", fontsize=6.9, fontweight="bold", color=INK)
            y -= 1.0
            last = g
        a.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round")
        a.scatter(est, y, s=26, facecolor=col if filled else "white", edgecolor=col, linewidth=1.1, zorder=3)
        a.annotate(m(est, "{:.1f}", sign=True), xy=(hi, y), xytext=(4, 0), textcoords="offset points",
                   ha="left", va="center", fontsize=6.6, color=INK2)
        labels.append(lab); ypos.append(y); y -= 1.0
    a.set_yticks(ypos); a.set_yticklabels(labels); a.tick_params(axis="y", length=0, labelcolor=INK2, labelsize=6.8)
    a.spines["left"].set_visible(False)
    a.axvline(0, color=INK2, lw=0.7)
    a.set_xlim(-5, 25); a.set_xticks([0, 5, 10, 15, 20]); a.set_ylim(y + 0.4, 0.6)
    a.set_xlabel("Difference in success (pp, 95% CI)")
    a.xaxis.grid(True); a.set_axisbelow(True)
    panel_at(a, "a", "Success differences (Reacher unless noted)", xfig=0.0)

    x = np.arange(3)
    off = {"Official": -0.14, "Refit (mean)": 0.14}
    for name, (single, real) in STREAMS.items():
        xs = x + off[name]
        for xi, s1, r1 in zip(xs, single, real):
            b.plot([xi, xi], [s1, r1], color=HAIR, lw=1.2, zorder=1)
        mk = "o" if name == "Official" else "s"
        b.scatter(xs, single, s=20, marker=mk, facecolor="white", edgecolor=OFFICIAL, linewidth=1.0, zorder=3)
        b.scatter(xs, real, s=20, marker=mk, color=HIST, zorder=3)
    b.set_xticks(x); b.set_xticklabels(["A", "B", "C"]); b.set_xlabel("Planner stream")
    b.set_ylim(60, 102); b.set_yticks([60, 70, 80, 90, 100]); b.set_ylabel("Reacher success (%)")
    b.yaxis.grid(True); b.set_axisbelow(True); b.set_xlim(-0.5, 2.5)
    hd = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=OFFICIAL, ms=4.0, label="single frame"),
          plt.Line2D([], [], marker="o", ls="", color=HIST, ms=4.0, label="real history"),
          plt.Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, ms=3.8, label="official"),
          plt.Line2D([], [], marker="s", ls="", mfc="none", mec=MUTED, ms=3.6, label="refit mean")]
    b.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, columnspacing=0.9,
             handletextpad=0.2, borderaxespad=0.0)
    panel_at(b, "b", "Stream spread", xfig=0.0)

    # Panel c: preregistered 2 x 2 budget-by-offset history gains (E1), with the
    # corrected budget contrasts (E2) read from the R8-FIX table.
    e1 = {25: {50: BUDGET_POINTS[0][1:4], 100: BUDGET_POINTS[2][1:4]},
          50: {50: BUDGET_POINTS[1][1:4], 100: BUDGET_POINTS[3][1:4]}}
    e2 = {lab.split()[1]: (est, lo, hi) for lab, est, lo, hi, kind in BUDGET_POINTS if kind == "e2"}
    for offv, col, marker, dodge in [(25, HIST, "o", -1.5), (50, CTX, "s", 1.5)]:
        xs = np.array([50, 100]) + dodge
        means = np.array([e1[offv][bu][0] for bu in (50, 100)])
        lows = np.array([e1[offv][bu][1] for bu in (50, 100)])
        highs = np.array([e1[offv][bu][2] for bu in (50, 100)])
        c.errorbar(xs, means, yerr=[means - lows, highs - means], fmt="none", ecolor=col, elinewidth=0.9, capsize=2)
        c.plot(xs, means, color=col, lw=1.2, marker=marker, ms=4.2, mfc="white", mec=col, label=f"offset {offv}")
    c.axhline(0, color=INK2, lw=0.6)
    c.set_xlabel("Budget (raw steps)"); c.set_ylabel("History gain (pp)")
    c.set_xticks([50, 100]); c.set_xlim(40, 110)
    c.set_ylim(-2, 20); c.set_yticks([0, 5, 10, 15]); c.yaxis.grid(True); c.set_axisbelow(True)
    c.legend(loc="lower left", bbox_to_anchor=(0.01, 0.15), handletextpad=0.3, borderaxespad=0.0)
    fmt_ci = lambda t: f"{m(t[0], '{:.1f}', True)} [{m(t[1], '{:.1f}')}, {m(t[2], '{:.1f}')}]"
    c.text(0.98, 0.97, "Budget 50 − budget 100", transform=c.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
    c.text(0.98, 0.86, "offset 25: " + fmt_ci(e2["25"]), transform=c.transAxes, ha="right", va="top", fontsize=6.5, color=HIST)
    c.text(0.98, 0.75, "offset 50: " + fmt_ci(e2["50"]), transform=c.transAxes, ha="right", va="top", fontsize=6.5, color=CTX)
    panel_at(c, "c", "Budget × target offset", xfig=0.455)
    save(fig, "fig4_history")


# ---------------------------------------------------------------- Figure 5
def fig5():
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.70), gridspec_kw={"wspace": 0.85, "width_ratios": [0.82, 0.82, 0.82, 1.20]})
    short = ["Selection loss", "Rank correlation", "Success (%)", "Scoring success (%)"]
    for i, (ax, (ttl, sub, v0, v1, d, lo, hi, fmt, labs)) in enumerate(zip(axes, PUSHT)):
        if i == 3:
            # M4 is a success-rate comparison on the same 100 cases pooled
            # across three planner streams.  Draw the three rates and place
            # the pairwise deltas underneath; do not treat the deltas as the
            # y-axis (the previous panel did that incorrectly).
            rates = [v0, v1, d]
            labels = ["latent", "sim-latent", "sim-task"]
            x = np.arange(3)
            ax.bar(x, rates, color=["#DAD9D3", "#55544F", INK], edgecolor=[OFFICIAL, "#55544F", INK], linewidth=0.8, width=0.65)
            for xx, rr in zip(x, rates):
                ax.text(xx, rr + 1.0, f"{rr:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5, linespacing=0.9)
            ax.set_ylim(0, 108); ax.set_ylabel("Success (%)")
            ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=6.5)
            for k, (arm, name) in enumerate([("SIM_LAT_RERANK", "sim-latent − latent"), ("SIM_TASK_RERANK", "sim-task − latent")]):
                dd, dlo, dhi = M4_DELTA[arm]
                ax.annotate(f"{name}\n{m(dd, '{:.1f}', True)} pp [{m(dlo, '{:.1f}', True)}, {m(dhi, '{:.1f}', True)}]",
                            xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -27 - 21 * k), textcoords="offset points",
                            ha="center", va="top", fontsize=6.5, color=INK2)
            panel(ax, chr(97 + i), short[i], dx=-14, dy=13)
            ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(-14 + 9.5, 4), textcoords="offset points",
                        fontsize=6.5, color=MUTED, ha="left", va="bottom")
            continue
        col = REFIT if i < 3 else INK
        ax.plot([0, 1], [v0, v1], color=HAIR, lw=1.2, zorder=1)
        ax.scatter([0], [v0], s=24, facecolor="white", edgecolor=OFFICIAL, linewidth=1.0, zorder=3)
        ax.scatter([1], [v1], s=24, color=col, zorder=3)
        ax.annotate(fmt.format(v0), xy=(0, v0), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.6, color=INK2)
        down = v1 < v0
        ax.annotate(fmt.format(v1), xy=(1, v1), xytext=(0, -6 if down else 5), textcoords="offset points",
                    ha="center", va="top" if down else "bottom", fontsize=6.6, color=INK2)
        ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1])
        ax.set_xticklabels(labs, fontsize=6.6)
        span = abs(v1 - v0)
        pad = max(span * 0.8, 0.02 if v0 < 1.5 else 3)
        top = max(v0, v1) + pad
        ax.set_ylim(min(v0, v1) - pad, min(top, 103) if v0 > 1.5 else top)
        ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=6.5)
        if fmt == "{:.3f}":
            txt = f"Δ {m(d, '{:.3f}', True)}\n[{m(lo, '{:.3f}', True)}, {m(hi, '{:.3f}', True)}]"
        else:
            txt = f"Δ {m(d, '{:.1f}', True)} pp\n[{m(lo, '{:.1f}', True)}, {m(hi, '{:.1f}', True)}]"
        ax.annotate(txt, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -27), textcoords="offset points",
                    ha="center", va="top", fontsize=6.5, color=INK2)
        panel(ax, chr(97 + i), short[i], dx=-14, dy=13)
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(-14 + 9.5, 4), textcoords="offset points",
                    fontsize=6.5, color=MUTED, ha="left", va="bottom")
    save(fig, "fig5_pusht_ranking")


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5()
    src = [FT_PATH, H3_PATH, PROBE_PATH, RH_PATH, ATTR_PATH, R6_RAW_PATH, R6_STATS_PATH, R8_MAIN_PATH, R8_SECONDARY_PATH, R8_E2_FIX_PATH, M4_MAIN_PATH, M4_SECONDARY_PATH, M4_RAW_PATH, W6_PATH, R10_PATH]
    lines = ["# Figure source registration", "", "Deterministic plots from sealed local evidence; no new experiments.", ""]
    for p in src:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"- `{p.relative_to(ROOT)}` — SHA256 `{h}`")
    (OUT / "FIGURE_SOURCES.md").write_text("\n".join(lines) + "\n")
    print("done")
