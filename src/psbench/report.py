"""Collect result tables into a markdown summary and a few figures."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics

_EFFECT_RE = re.compile(r"effects_(.+?)_(nb_glm|mixed|welch|mixscape)_chunk\d+\.csv$")
_NULL_RE = re.compile(r"null_(.+?)_(nb_glm|mixed|welch)\.csv$")


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _collect(results: Path, regex: re.Pattern) -> dict[str, pd.DataFrame]:
    out: dict[str, list] = {}
    for f in sorted(results.glob("*.csv")):
        m = regex.search(f.name)
        if m:
            df = _read(f)
            if len(df):
                out.setdefault(m.group(1), []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def _section(lines: list[str], title: str, df: pd.DataFrame, floatfmt: str = ".3f") -> None:
    lines += [f"### {title}", "", df.to_markdown(index=False, floatfmt=floatfmt), ""]


def build_report(results_dir: str, out_dir: str, fdr: float = 0.05) -> Path:
    results, out = Path(results_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Perturb-seq benchmark summary", ""]

    guides = results / "guide_benchmark.csv"
    if guides.exists():
        g = pd.read_csv(guides)
        lines += ["## Guide assignment (simulated ground truth)", ""]
        _section(lines, "Mean F1 by method and signal strength",
                 g.pivot_table(index="method", columns="signal_mean", values="f1", aggfunc="mean").reset_index())
        mois = sorted(g["moi"].unique())
        fig, axes = plt.subplots(1, len(mois), figsize=(4 * len(mois), 3.2), sharey=True, squeeze=False)
        for ax, moi in zip(axes[0], mois):
            for method, d in g[g["moi"] == moi].groupby("method"):
                ax.plot(d["signal_mean"], d["f1"], marker="o", label=method)
            ax.set_xscale("log")
            ax.set_title(f"MOI {moi}")
            ax.set_xlabel("mean signal UMI")
        axes[0][0].set_ylabel("F1")
        axes[0][-1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "guide_assignment_f1.png", dpi=150)
        plt.close(fig)

    effects = _collect(results, _EFFECT_RE)
    for ds, res in effects.items():
        lines += [f"## Effect methods: {ds}", ""]
        _section(lines, "On-target knockdown", metrics.on_target_summary(res, fdr))
        _section(lines, "DEGs per perturbation (FDR < %.2f)" % fdr, metrics.deg_summary(res, fdr))
    for f in sorted(results.glob("mixscape_*_summary.csv")):
        m = _read(f)
        if len(m):
            _section(lines, f"Mixscape KO fraction ({f.stem})", m[["ko_fraction"]].describe().reset_index())

    nulls = _collect(results, _NULL_RE)
    for ds, df in nulls.items():
        agg = df.groupby("method")[["frac_p_lt_0.05", "n_deg_fdr_0.05", "ks_vs_uniform"]].mean().reset_index()
        lines += [f"## Null calibration against non-targeting cells: {ds}", ""]
        _section(lines, "Mean over pseudo-perturbations (expected frac_p_lt_0.05 = 0.05, n_deg = 0)", agg)

    for f in sorted(results.glob("power_*.csv")):
        if f.stem.endswith("_downsample"):
            continue
        p = _read(f)
        if not len(p):
            continue
        ds = f.stem.removeprefix("power_")
        lines += [f"## Power: {ds}", ""]
        _section(lines, "Mean power (rows: perturbed cells, columns: log2FC)",
                 p.pivot_table(index="n_cells", columns="log2fc", values="power", aggfunc="mean").reset_index())
        _section(lines, "Observed FDR (target 0.05)",
                 p.pivot_table(index="n_cells", columns="log2fc", values="observed_fdr", aggfunc="mean").reset_index())
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for lfc, d in p.groupby("log2fc"):
            d = d.groupby("n_cells")["power"].mean()
            ax.plot(d.index, d.values, marker="o", label=f"log2FC {lfc}")
        ax.set_xscale("log")
        ax.set_xlabel("perturbed cells")
        ax.set_ylabel("power")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"power_{ds}.png", dpi=150)
        plt.close(fig)

    for f in sorted(results.glob("interaction_*_summary.csv")):
        s = _read(f)
        if not len(s):
            continue
        ds = f.stem.removeprefix("interaction_").removesuffix("_summary")
        lines += [f"## Interactions: {ds}", ""]
        _section(lines, "Combinations by class", s["class"].value_counts().rename_axis("class").reset_index())
        _section(lines, "Summary", s[["additive_error", "magnitude_ratio", "n_interaction_genes"]].describe().reset_index())

    for f in sorted(results.glob("model_*_cv.csv")):
        cv = _read(f)
        if not len(cv):
            continue
        ds = f.stem.removeprefix("model_").removesuffix("_cv")
        lines += [f"## Double-perturbation prediction: {ds}", ""]
        agg = cv.groupby("model")[["mse", "pearson", "mse_top", "pearson_top"]].mean().reset_index()
        _section(lines, "Cross-validated metrics on held-out doubles", agg.sort_values("mse"), ".4f")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.bar(agg["model"], agg["pearson_top"])
        ax.set_ylabel("Pearson, top 20 genes")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out / f"model_{ds}.png", dpi=150)
        plt.close(fig)

    path = out / "summary.md"
    path.write_text("\n".join(lines))
    return path
