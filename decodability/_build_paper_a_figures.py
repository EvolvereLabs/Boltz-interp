"""Generator for paper_a_figures.ipynb  (NeurIPS LMRL workshop — Paper A).

Builds the 5 main figures (R1-R5 of PAPER_A_OUTLINE.md) from the benchmark
aggregates already staged locally. Publication style; every figure saved as
both .pdf (vector, editable text) and .png into figs_paper_a/.

Data sources (all local, no network):
  - df (per-concept benchmark)  : sae_explore/layer_sweep_rec{0,1}/benchmark/layer{L}/*_agg.json
                                  sae_explore/diffusion_analysis/rec{R}/benchmark/layer{L}/*_agg.json
  - amino-acid identity floor   : sae_explore/amino_acid_sanity/results.jsonl
  - precision / recall          : sae_explore/precision_recall{,_diffusion}.csv
  - SS label source (R5)        : {secondary,boltz_secondary}_seed{1,2,3}_benchmark.json per layer dir

Run:  python _build_paper_a_figures.py   (then execute the notebook)

Figure -> subsection map:
  fig1  R1  amino-acid identity calibration (the instrument works)
  fig2  R2  the trunk encodes geometry AND chemistry
  fig3  R3  geometry retained, chemistry discarded  (2-panel hero)
  fig4  R4  probe > SAE ~ neuron, all recall-leaning
  fig5  R5  under-annotation inflates apparent false positives
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------- intro
md(r"""# Paper A — main figures (NeurIPS LMRL workshop)

**Thesis.** Boltz-1's pairformer *trunk* builds a representation of both geometry **and**
sequence-chemistry; its *diffusion* module represents geometry alone — refining structure
rather than creating it, and progressively discarding chemical/sequence information, down to
amino-acid identity itself.

This notebook produces the five main figures. It is **generated** by
`_build_paper_a_figures.py` — edit the generator, not the .ipynb (direct edits are wiped on
regen). Each figure is saved to `figs_paper_a/` as `.pdf` (vector) + `.png`.

| fig | subsection | claim |
|---|---|---|
| 1 | R1 | the three readouts behave as expected on residue identity → instrument calibrated |
| 2 | R2 | trunk probes recover geometry **and** chemistry concepts |
| 3 | R3 | diffusion keeps geometry, discards chemistry (depth-matched + time-course) |
| 4 | R4 | probe ≫ SAE ≈ neuron; every readout is recall-leaning |
| 5 | R5 | apparent false positives are partly SwissProt under-annotation |

Metric reminder: **probe-raw** = held-out L2 probe on raw activations (SAE-independent ceiling);
**SAE-1feat / neuron-1feat** = best single latent / neuron (in-sample). Headline claims lead with
probe-raw.""")

# ----------------------------------------------------------------- setup
code(r"""import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- publication style ----
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # keep text editable in vector output
})

ROOT = Path.cwd()
DATA = ROOT / "data"   # committed summary data (agg JSONs + CSVs)
PF_DIRS  = {1: DATA/"sae_explore/layer_sweep_rec1/benchmark"}
DIFF_DIR = DATA/"sae_explore/diffusion_analysis"
FIG_DIR  = ROOT/"figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

MODULE_MAX = {"pairformer": 47, "diffusion": 22}   # diffusion module ~half as deep
PF_REC, DIFF_REC = 1, 199                           # each stack's most-formed setting

# palette: stacks + concept families
C_PF, C_DIFF = "#1f77b4", "#d62728"                 # pairformer / diffusion
C_GEOM, C_CHEM = "#1f77b4", "#d62728"               # geometry / chemistry families
C_PROBE, C_SAE, C_NEUR = "#2ca02c", "#1f77b4", "#ff7f0e"

SS = ["secondary_structure:helix", "secondary_structure:strand", "secondary_structure:coil"]

# SwissProt's own experimental secondary-structure annotations duplicate the (dense, reliable)
# DSSP `secondary_structure:*` track but are sparse/under-annotated, so their absolute F1 is not a
# clean readout of "what the model knows". We classify them as geometry (they ARE structural) but
# EXCLUDE them from the family figures (Fig 2 / Fig 3A) to avoid a redundant, confusing second
# "helix"/"strand"; the DSSP track represents geometry there. Documented, not silent.
STRUCT_DUP = {"helix", "beta_strand", "turn", "coiled_coil"}

def family(c):                       # geometry/structure vs sequence-chemistry
    if (c.startswith("secondary_structure:") or "disordered" in c
            or "coiled_coil" in c or c in STRUCT_DUP):
        return "geometry"
    return "chemistry"

def short(c):
    return (c.replace("secondary_structure:", "ss:").replace("compositional_bias:", "cb:")
             .replace("modified_residue:", "mod:").replace("region:", ""))

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR/f"{name}.{ext}")
    print("saved", (FIG_DIR/f"{name}.png").relative_to(ROOT), "(+ .pdf)")

print("pairformer dirs:", {r: d.exists() for r, d in PF_DIRS.items()}, "| diffusion:", DIFF_DIR.exists())""")

# ----------------------------------------------------------------- loaders
md(r"""## Load the benchmark into one tidy table

Same loader as `pairformer_vs_diffusion_analysis.ipynb`: flatten every
`(stack, rec, layer, concept_set, concept)` `*_agg.json` into one long DataFrame, then derive
the informativeness filter (drop concepts that are at probe-chance in **both** stacks).""")

code(r"""def _flatten_agg(path, stack):
    d = json.load(open(path)); rec, layer, cset = d["rec"], d["layer"], d["concept_set"]
    return [dict(stack=stack, rec=rec, layer=layer, concept_set=cset, concept=concept,
                 sae_f1=c.get("sae_f1_mean"), sae_null=c.get("sae_null_mean"),
                 sae_sig=c.get("sae_sig_mean"), neuron_f1=c.get("neuron_f1"),
                 neuron_null=c.get("neuron_null_mean"),
                 probe_raw_f1=c.get("probe_raw_f1"), probe_sae_f1=c.get("probe_sae_f1_mean"),
                 n_proteins=d.get("n_proteins"))
            for concept, c in d["per_concept"].items()]

def load_concepts():
    rows = []
    for rec, bench in PF_DIRS.items():
        for p in glob.glob(str(bench/"layer*/[sw]*_agg.json")):
            rows += _flatten_agg(p, "pairformer")
    for p in glob.glob(str(DIFF_DIR/"rec*/benchmark/layer*/[sw]*_agg.json")):
        rows += _flatten_agg(p, "diffusion")
    return pd.DataFrame(rows).drop_duplicates(["stack", "rec", "layer", "concept_set", "concept"])

df = load_concepts()
df["frac_depth"] = df["layer"] / df["stack"].map(MODULE_MAX)
df["neuron_sig"] = df["neuron_f1"] - df["neuron_null"]
df["probe_above_null"] = df["probe_raw_f1"] - df["sae_null"]

# informativeness: keep a concept if held-out probe-raw clears chance by >0.05 in EITHER stack
_by = df.groupby(["concept", "stack"])["probe_above_null"].median().unstack("stack")
INFO_CONCEPTS = sorted(_by.index[_by.max(axis=1) > 0.05])
DROPPED = sorted(set(df["concept"]) - set(INFO_CONCEPTS))

def end_of_module(stack, rec):
    s = df[(df["stack"] == stack) & (df.rec == rec)]
    return s[s.layer == s.layer.max()].set_index("concept")

print(f"concept rows: {len(df)} | informative: {len(INFO_CONCEPTS)}/{df.concept.nunique()}")
print("dropped (probe~chance both stacks):", DROPPED)
print("pairformer layers:", sorted(df[df["stack"]=='pairformer'].layer.unique()))
print("diffusion layers :", sorted(df[df["stack"]=='diffusion'].layer.unique()),
      "| recs:", sorted(df[df["stack"]=='diffusion'].rec.unique()))""")

# ================================================================= FIG 1 (R1)
md(r"""## Figure 1 (R1) — Amino-acid identity calibrates the instrument

The most local, model-visible concept: which of the 20 amino acids a residue is. In the trunk
the **probe is pinned at ~1.0** (identity is the model's input, fully linearly present — confirms
label/activation alignment), the best single **SAE latent concentrates it monosemantically**
(~0.85), and the best **raw neuron is weak** (~0.54). This fixes the F1 reference scale every
later figure is read against.""")

code(r"""aa = pd.DataFrame([json.loads(l) for l in
                   (DATA/"sae_explore/amino_acid_sanity/results.jsonl").read_text().splitlines() if l.strip()])
aa_pf = (aa[aa.layer_type == "pairformer"]
         .groupby("layer")[["probe_f1", "sae_f1", "neuron_f1"]].agg(["mean", "std"]))
aa_pf.columns = ["_".join(c) for c in aa_pf.columns]
aa_pf["frac_depth"] = aa_pf.index / MODULE_MAX["pairformer"]
aa_pf = aa_pf.sort_values("frac_depth")

fig, ax = plt.subplots(figsize=(6.2, 4.4))
for col, c, lab in [("probe_f1", C_PROBE, "linear probe (full vector)"),
                    ("sae_f1", C_SAE, "best single SAE latent"),
                    ("neuron_f1", C_NEUR, "best single neuron")]:
    m, s = aa_pf[f"{col}_mean"], aa_pf[f"{col}_std"]
    ax.plot(aa_pf.frac_depth, m, "o-", color=c, label=lab, ms=4)
    ax.fill_between(aa_pf.frac_depth, m - s, m + s, color=c, alpha=0.13)
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_xlabel("fractional depth  (layer / module depth)")
ax.set_ylabel("amino-acid identity F1  (mean ± sd over 20 AA)")
ax.set_ylim(0, 1.05); ax.set_xlim(-0.02, 1.02)
ax.set_title("Pairformer trunk: residue identity is fully decodable")
ax.legend(loc="lower right")
plt.tight_layout(); save(fig, "fig1_aa_calibration"); plt.show()

print("trunk output (L47):")
print(aa[(aa.layer_type=='pairformer') & (aa.layer==47)][["probe_f1","sae_f1","neuron_f1"]].mean().round(3))""")

# ================================================================= FIG 2 (R2)
md(r"""## Figure 2 (R2) — The trunk encodes geometry *and* chemistry

SAE-independent linear-probe ceiling (`probe-raw`) vs relative depth in the pairformer trunk
(rec 1), one thin line per informative concept, coloured by family (blue = geometry/structure,
red = sequence-chemistry). Bold lines are the family means. Both families rise into the decodable
range — the trunk is not geometry-only. The disulfide-bond curve shows a distinct **mid-trunk
hotspot** (annotated).""")

code(r"""fam_concepts = [c for c in INFO_CONCEPTS if c not in STRUCT_DUP]   # drop redundant SwissProt SS dups
print("family figures use", len(fam_concepts), "concepts; excluded SwissProt SS duplicates:",
      sorted(set(INFO_CONCEPTS) & STRUCT_DUP))
sub = df[(df["stack"] == "pairformer") & (df.rec == PF_REC) & (df.concept.isin(fam_concepts))]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for concept, g in sub.groupby("concept"):
    g = g.sort_values("frac_depth")
    fam = family(concept)
    ax.plot(g.frac_depth, g.probe_raw_f1, "-", color=(C_GEOM if fam == "geometry" else C_CHEM),
            alpha=0.22, lw=1.2)
# family means (bold)
for fam, col in [("geometry", C_GEOM), ("chemistry", C_CHEM)]:
    cs = [c for c in sub.concept.unique() if family(c) == fam]
    g = (sub[sub.concept.isin(cs)].groupby("frac_depth").probe_raw_f1.mean().sort_index())
    ax.plot(g.index, g.values, "o-", color=col, lw=2.6, ms=5, label=f"{fam} mean ({len(cs)} concepts)")
# highlight disulfide hotspot
ds = sub[sub.concept == "disulfide_bond"].sort_values("frac_depth")
if len(ds):
    ax.plot(ds.frac_depth, ds.probe_raw_f1, "s-", color="#8c1d04", lw=1.6, ms=4, label="disulfide bond")
    pk = ds.loc[ds.probe_raw_f1.idxmax()]
    ax.annotate("mid-trunk\ndisulfide hotspot", (pk.frac_depth, pk.probe_raw_f1),
                xytext=(pk.frac_depth + 0.12, pk.probe_raw_f1 + 0.18), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#8c1d04", lw=1.2))
ax.set_xlabel("fractional depth  (layer / module depth)")
ax.set_ylabel("probe-raw F1  (held-out)")
ax.set_ylim(0, 1.0); ax.set_xlim(-0.02, 1.02)
ax.set_title("Pairformer trunk encodes both families of concept")
ax.legend(loc="upper left", framealpha=0.9)
plt.tight_layout(); save(fig, "fig2_trunk_geom_chem"); plt.show()""")

# ================================================================= FIG 3 (R3)
md(r"""## Figure 3 (R3) — Geometry retained, chemistry discarded  *(headline)*

**3A — depth-matched module output.** probe-raw F1 at each module's final layer (pairformer L47 =
trunk output, diffusion L22 = module output), concepts grouped by family. Geometry bars are
near-equal; every chemistry bar collapses in diffusion. Amino-acid identity (the purest sequence
concept) is the extreme. Both stacks' bars carry **95% by-protein bootstrap CIs** (2000 resamples;
`bootstrap_concept_f1.py`) — the geometry/chemistry gap exceeds the intervals even for the rare
labels (full per-concept forest plot in supplementary S10). The cross-seed repeats cannot supply this
interval: probe-raw is deterministic across seeds, so the uncertainty here is finite-sample, estimated
by resampling proteins.

**3B, 3C — diffusion layer × step grids.** probe-raw F1 over the diffusion module's depth (layer)
and denoising time (diffusion step), for one geometry concept (helix) and one chemistry concept
(signal peptide). Geometry is bright across the *entire* grid — already resolved at the first step
and at every layer (the module refines, it does not create) — while chemistry is dark everywhere:
the diffusion module never carries it, at any depth or step.""")

code(r"""# ----- 3A: depth-matched bars -----
pf_e, di_e = end_of_module("pairformer", PF_REC), end_of_module("diffusion", DIFF_REC)
comp = pd.DataFrame({"pairformer": pf_e["probe_raw_f1"], "diffusion": di_e["probe_raw_f1"]}).dropna()
comp = comp[comp.index.isin(INFO_CONCEPTS) & ~comp.index.isin(STRUCT_DUP)]   # drop redundant SS dups
aa_pf47 = aa[(aa.layer_type=='pairformer') & (aa.layer==47)]["probe_f1"].mean()
aa_di22 = aa[(aa.layer_type=='diffusion')  & (aa.layer==22)]["probe_f1"].mean()
comp.loc["aa:identity"] = [aa_pf47, aa_di22]
comp["fam"] = [("geometry" if c != "aa:identity" and family(c) == "geometry" else "chemistry")
               for c in comp.index]
comp["_o"] = comp.fam.map({"geometry": 0, "chemistry": 1})     # geometry at bottom band
comp = comp.sort_values(["_o", "pairformer"], ascending=[True, True])

# 95% by-protein bootstrap CIs for the depth-matched module-output bars (bootstrap_concept_f1.py):
# pairformer L47 (trunk output) and diffusion L22 (module output). These quantify the small/uneven-
# eval-set uncertainty the reviewer flagged; seeds cannot (probe-raw is deterministic across seeds).
_ci_path = DATA/"sae_explore/concept_f1_ci.csv"
PF_CI, DIFF_CI = {}, {}
if _ci_path.exists():
    _ci = pd.read_csv(_ci_path)
    _ci = _ci[_ci.method == "probe_raw"]
    PF_CI = {r.concept: (r.f1_lo, r.f1_hi)
             for r in _ci[(_ci.layer == 47) & (_ci.stack == "pairformer")].itertuples()}
    DIFF_CI = {r.concept: (r.f1_lo, r.f1_hi)
               for r in _ci[(_ci.layer == 22) & (_ci.stack == "diffusion")].itertuples()}

# ----- 3B/3C: diffusion layer × step grids (section-7 style) -----
steps_d = [0, 10, 50, 100, 199]; layers_d = [0, 2, 4, 6, 10, 14, 18, 22]
def _diff_grid(concept):
    s = df[(df["stack"] == "diffusion") & (df.concept == concept)]
    return s.pivot_table(index="rec", columns="layer", values="probe_raw_f1").reindex(index=steps_d, columns=layers_d)

# Font sizes tuned so the figure is legible at 100% zoom when placed at \textwidth.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, FS_SIDE, FS_CELL = 13, 12, 11, 10, 11, 9
# A on the LEFT at half width (spanning both rows); B (top) and C (bottom) stacked on the RIGHT,
# with a dedicated thin colorbar column on the far right so nothing overlaps.
fig, axd = plt.subplot_mosaic([["A", "B", ".", "cbar"], ["A", "C", ".", "cbar"]], figsize=(12, 6.6),
                              gridspec_kw={"width_ratios": [1.0, 1.0, 0.16, 0.045]}, layout="constrained")
fig.get_layout_engine().set(w_pad=0.10, h_pad=0.08, hspace=0.06, wspace=0.06)
# A: depth-matched bars
ax = axd["A"]; y = np.arange(len(comp)); h = 0.38
# asymmetric x-error from the bootstrap CI, centred on each bar's plotted value (0 where no CI, e.g. aa:identity)
def _xerr(series, ci):   # asymmetric (lo, hi) error from the bootstrap CI, centred on each bar's value
    lo = np.array([max(0.0, v - ci[c][0]) if c in ci else 0.0 for c, v in series.items()])
    hi = np.array([max(0.0, ci[c][1] - v) if c in ci else 0.0 for c, v in series.items()])
    return np.vstack([lo, hi])
_ekw = dict(elinewidth=1.8, ecolor="k", capsize=4, capthick=1.8)
ax.barh(y + h/2, comp["pairformer"], h, color=C_PF, edgecolor="k", linewidth=0.4,
        xerr=_xerr(comp["pairformer"], PF_CI), error_kw=_ekw,
        label="pairformer L47 (95% CI)")
ax.barh(y - h/2, comp["diffusion"], h, color=C_DIFF, edgecolor="k", linewidth=0.4,
        xerr=_xerr(comp["diffusion"], DIFF_CI), error_kw=_ekw,
        label="diffusion L22 (95% CI)")
ax.set_yticks(y); ax.set_yticklabels([short(c) for c in comp.index], fontsize=FS_TICK)
n_geom = (comp.fam == "geometry").sum()
ax.axhline(n_geom - 0.5, color="k", lw=0.8, ls="--")
ax.text(0.99, n_geom/2 - 0.5, "geometry", rotation=90, va="center", ha="right",
        transform=ax.get_yaxis_transform(), fontsize=FS_SIDE, color=C_GEOM, alpha=0.7)
ax.text(0.99, (n_geom + len(comp))/2 - 0.5, "chemistry", rotation=90, va="center", ha="right",
        transform=ax.get_yaxis_transform(), fontsize=FS_SIDE, color=C_CHEM, alpha=0.7)
ax.set_xlabel("probe-raw F1  (held-out)", fontsize=FS_LABEL); ax.set_xlim(0, 1.0)
ax.tick_params(axis="x", labelsize=FS_TICK)
ax.set_title("(A) Module output, depth-matched", fontsize=FS_TITLE)
# Legend off the bars, in the empty right-hand gap beside the short chemistry rows.
ax.legend(loc="center right", bbox_to_anchor=(0.995, 0.55), fontsize=FS_LEG, framealpha=0.9)

# B/C: diffusion layer × step heatmaps
im = None
for key, concept, lab in [("B", "secondary_structure:helix", "(B) helix — geometry, retained everywhere"),
                          ("C", "signal_peptide", "(C) signal peptide — sequence, progressively lost")]:
    ax = axd[key]; g = _diff_grid(concept)
    im = ax.imshow(g.values, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(layers_d))); ax.set_xticklabels(layers_d, fontsize=FS_TICK)
    ax.set_yticks(range(len(steps_d)));  ax.set_yticklabels(steps_d, fontsize=FS_TICK)
    ax.set_xlabel("diffusion layer  (depth →)", fontsize=FS_LABEL)
    if key == "B": ax.set_ylabel("diffusion step\n(0=noisy → 199=final)", fontsize=FS_LABEL)
    ax.set_title(lab, fontsize=FS_TITLE); ax.grid(False)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                       color="w" if v < 0.6 else "k", fontsize=FS_CELL)
cb = fig.colorbar(im, cax=axd["cbar"])
cb.set_label("probe-raw F1", fontsize=FS_LABEL); cb.ax.tick_params(labelsize=FS_TICK)
save(fig, "fig3_geom_retained_chem_discarded"); plt.show()

print("== 3A depth-matched probe-raw F1 ==")
print(comp.assign(gap=comp.pairformer - comp.diffusion).round(3).to_string())""")

# ================================================================= FIG 4 (R4)
md(r"""## Figure 4 (R4) — Probe ≫ SAE ≈ neuron, and every readout is recall-leaning

Real precision/recall (recomputed by `compute_precision_recall.py`) for the three readouts, both
stacks, informative concepts only. Every point sits **below the diagonal** (recall > precision):
all readouts behave as broad, somewhat noisy detectors. The **probe** cluster dominates;
**SAE-1feat and neuron-1feat overlap** — the SAE's edge over raw neurons is small and trunk-only.""")

code(r"""frames = [pd.read_csv(p) for p in
          [DATA/"sae_explore/precision_recall.csv", DATA/"sae_explore/precision_recall_diffusion.csv"]
          if p.exists()]
pr = pd.concat(frames, ignore_index=True)
pr = pr[pr.concept.isin(INFO_CONCEPTS)]
LAB = {"sae": "SAE-1feat", "neuron": "neuron-1feat", "probe_raw": "probe-raw"}
MK  = {"sae": "o", "neuron": "^", "probe_raw": "s"}
MC  = {"sae": C_SAE, "neuron": C_NEUR, "probe_raw": C_PROBE}
stacks = [s for s in ["pairformer", "diffusion"] if s in pr["stack"].unique()]

# by-protein bootstrap CIs (bootstrap_concept_f1.py) keyed by (stack, layer, concept, method); used as
# faint recall/precision whiskers on the points that have them (trunk). Quantifies the small-eval-set
# uncertainty per point without re-fitting; absent keys (e.g. diffusion) just get no whisker.
_cf = DATA/"sae_explore/concept_f1_ci.csv"
CI4 = {}
if _cf.exists():
    for r in pd.read_csv(_cf).itertuples():
        CI4[(r.stack, r.layer, r.concept, r.method)] = (r.recall_lo, r.recall_hi, r.precision_lo, r.precision_hi)

fig, axes = plt.subplots(1, len(stacks), figsize=(6.2*len(stacks), 5.4), squeeze=False, sharex=True, sharey=True)
for ax, stack in zip(axes[0], stacks):
    for m in ["probe_raw", "sae", "neuron"]:
        g = pr[(pr.method == m) & (pr["stack"] == stack)]
        for row in g.itertuples():
            k = (stack, row.layer, row.concept, m)
            if k in CI4:
                rl, rh, pl, ph = CI4[k]
                ax.plot([rl, rh], [row.precision, row.precision], color=MC[m], lw=0.7, alpha=0.3, zorder=1)
                ax.plot([row.recall, row.recall], [pl, ph], color=MC[m], lw=0.7, alpha=0.3, zorder=1)
        ax.scatter(g.recall, g.precision, s=55, marker=MK[m], c=MC[m],
                   edgecolor="k", linewidth=0.3, alpha=0.8, label=LAB[m], zorder=3)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("recall (per-domain)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(stack); ax.legend(loc="lower right")
axes[0][0].set_ylabel("precision (per-residue)")
fig.suptitle("Precision vs recall — below the diagonal = recall-leaning; probe dominates, SAE ≈ neuron", y=1.0)
plt.tight_layout(); save(fig, "fig4_precision_recall"); plt.show()

print("mean precision / recall / F1 by stack × method (informative concepts):")
print(pr.groupby(["stack", "method"])[["precision", "recall", "f1"]].mean().round(3))""")

# ================================================================= FIG 5 (R5)
md(r"""## Figure 5 (R5) — Apparent false positives are partly label incompleteness

The same physical concept can be labelled two ways. **DSSP** (run on the predicted structure) assigns
secondary structure to *every* residue; **SwissProt's own experimental annotations** (`helix`,
`beta_strand`) cover only residues backed by experimental evidence — most are left blank. Probing the
**identical** trunk activations, helix decodes at **~0.85** against dense DSSP labels but only **~0.42**
against SwissProt's sparse labels (strand 0.83 vs 0.33). The model plainly represents helices; the low
SwissProt F1 is **under-annotation**, not a decoder failure — so F1 against SwissProt is a precision
*lower bound*, which explains part of the recall>precision gap in Fig 4. (The complementary
self-consistency check — DSSP on Boltz's own vs AlphaFold's structure — is in supplementary S7.)""")

code(r"""# Same concept, two label sources: dense DSSP (secondary_structure:*) vs sparse SwissProt
# experimental annotations (helix / beta_strand). probe-raw F1, best over trunk layers, recycle 1.
r1 = df[(df["stack"] == "pairformer") & (df.rec == PF_REC)]
PAIRS = [("helix", "secondary_structure:helix", "helix"),
         ("strand", "secondary_structure:strand", "beta_strand")]
dssp = [r1[r1.concept == d].probe_raw_f1.max() for _, d, _ in PAIRS]
spr  = [r1[r1.concept == s].probe_raw_f1.max() for _, _, s in PAIRS]

fig, ax = plt.subplots(figsize=(6.6, 4.6))
x = np.arange(len(PAIRS)); w = 0.38
ax.bar(x - w/2, dssp, w, color=C_PF, edgecolor="k", linewidth=0.4,
       label="DSSP labels (dense — every residue)")
ax.bar(x + w/2, spr, w, color="#7f7f7f", hatch="//", edgecolor="k", linewidth=0.4,
       label="SwissProt experimental labels (sparse)")
for xi, (d, s) in enumerate(zip(dssp, spr)):
    ax.annotate(f"gap\n−{d-s:.2f}", (xi + w/2, s + 0.04), ha="center", va="bottom",
                fontsize=9, color="#8c1d04")
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in PAIRS])
ax.set_ylabel("probe-raw F1  (held-out, best over trunk layers)"); ax.set_ylim(0, 1.0)
ax.set_title("Same concept, different label completeness:\nSwissProt under-annotation depresses measured F1")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); save(fig, "fig5_under_annotation"); plt.show()
print("helix : DSSP", round(dssp[0], 3), "vs SwissProt", round(spr[0], 3))
print("strand: DSSP", round(dssp[1], 3), "vs SwissProt", round(spr[1], 3))""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = "paper_a_figures.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
