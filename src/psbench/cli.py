"""Command line entry point. Every Nextflow process calls one subcommand."""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

from . import data, effects, guides, interaction, metrics, model, power, report


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str, quick: bool = False) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    overrides = cfg.pop("quick", {})
    if quick:
        cfg = _merge(copy.deepcopy(cfg), overrides)
    for name, ds in cfg["datasets"].items():
        ds["name"] = name
    return cfg


def _write(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def cmd_inspect(a, cfg):
    adata = data.load_pertpy(cfg["datasets"][a.dataset]["loader"])
    print(data.describe(adata))


def cmd_download(a, cfg):
    adata = data.load_pertpy(cfg["datasets"][a.dataset]["loader"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(a.out)


def cmd_prepare(a, cfg):
    adata = ad.read_h5ad(a.input)
    prepared = data.prepare(adata, cfg["datasets"][a.dataset], cfg["preprocess"], cfg["seed"])
    prepared.write_h5ad(a.out)
    print(f"{prepared.n_obs} cells, {prepared.n_vars} genes, "
          f"{prepared.obs['perturbation'].nunique()} conditions")


def cmd_guides(a, cfg):
    _write(guides.benchmark(cfg["guides"], cfg["seed"]), a.out)
    if a.counts:
        counts = ad.read_h5ad(a.counts)
        calls = {name: fn(counts.X) for name, fn in guides.METHODS.items()}
        rows = [{"method": n, **guides.summarize_calls(c)} for n, c in calls.items()]
        _write(pd.DataFrame(rows), str(Path(a.out).with_name("guide_calls_real.csv")))
        _write(guides.pairwise_agreement(calls), str(Path(a.out).with_name("guide_agreement_real.csv")))


def cmd_effects(a, cfg):
    adata = ad.read_h5ad(a.input)
    ec = cfg["effects"]
    perts = effects.single_perturbations(adata)
    if ec.get("max_perts"):
        perts = perts[: ec["max_perts"]]
    perts = perts[a.chunk::a.n_chunks]
    res = effects.run_per_pert(adata, perts, a.method, **effects.method_kwargs(a.method, ec))
    _write(res, a.out)


def cmd_mixscape(a, cfg):
    adata = ad.read_h5ad(a.input)
    ec = cfg["effects"]
    perts = effects.single_perturbations(adata)
    if ec.get("max_perts"):
        perts = perts[: ec["max_perts"]]
    res, summary = effects.mixscape(adata, perts)
    _write(res, a.out)
    _write(summary, a.summary_out)


def cmd_null(a, cfg):
    adata = ad.read_h5ad(a.input)
    ec = cfg["effects"]
    res = metrics.null_calibration(
        adata, a.method, ec["null"]["n_groups"], ec["null"]["group_size"], cfg["seed"],
        **effects.method_kwargs(a.method, ec),
    )
    _write(res, a.out)


def cmd_power(a, cfg):
    adata = ad.read_h5ad(a.input)
    pc = cfg["power"]
    mu, alpha = power.estimate_params(adata)
    sim = power.simulate_power(
        mu, alpha, pc["n_cells"], pc["lfc"], pc["n_control"], pc["frac_de"], pc["n_rep"],
        fdr=cfg["effects"]["fdr"], seed=cfg["seed"],
    )
    _write(sim, a.out)
    ds = power.downsample_recovery(adata, pc["downsample_fractions"], seed=cfg["seed"])
    _write(ds, str(Path(a.out).with_name(Path(a.out).stem + "_downsample.csv")))


def cmd_interaction(a, cfg):
    adata = ad.read_h5ad(a.input)
    ic = cfg["interaction"]
    summary, genes = interaction.run_interactions(
        adata, min_cells=ic["min_cells"], fdr=ic["fdr"],
        synergy_ratio=ic["synergy_ratio"], buffer_ratio=ic["buffer_ratio"],
    )
    _write(summary, a.out)
    _write(genes, a.genes_out)


def cmd_model(a, cfg):
    adata = ad.read_h5ad(a.input)
    _write(model.run_cv(adata, cfg["model"], cfg["seed"]), a.out)


def cmd_report(a, cfg):
    print(report.build_report(a.results_dir, a.out, cfg["effects"]["fdr"]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psbench")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--quick", action="store_true", help="apply the `quick` overrides from the config")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, **flags):
        s = sub.add_parser(name)
        for flag, kw in flags.items():
            s.add_argument("--" + flag.replace("_", "-"), **kw)
        s.set_defaults(fn=fn)

    ds = {"required": True}
    add("inspect", cmd_inspect, dataset=ds)
    add("download", cmd_download, dataset=ds, out=ds)
    add("prepare", cmd_prepare, dataset=ds, input=ds, out=ds)
    add("guides", cmd_guides, out=ds, counts={"default": None})
    add("effects", cmd_effects, input=ds, out=ds, method={"required": True, "choices": list(effects.PER_PERT_METHODS)},
        chunk={"type": int, "default": 0}, n_chunks={"type": int, "default": 1})
    add("mixscape", cmd_mixscape, input=ds, out=ds, summary_out=ds)
    add("null", cmd_null, input=ds, out=ds, method={"required": True, "choices": list(effects.PER_PERT_METHODS)})
    add("power", cmd_power, input=ds, out=ds)
    add("interaction", cmd_interaction, input=ds, out=ds, genes_out=ds)
    add("model", cmd_model, input=ds, out=ds)
    add("report", cmd_report, results_dir=ds, out=ds)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.quick)
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
