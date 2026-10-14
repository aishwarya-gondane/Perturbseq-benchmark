# perturbseq-benchmark

A benchmark of guide-assignment and perturbation-effect methods on two public
Perturb-seq datasets, packaged as a Nextflow workflow with automated tests.

| Dataset | Cells | Perturbations | Used for |
|---|---|---|---|
| Replogle et al. 2022, K562 essential | single-gene CRISPRi | ~2,000 genes | effect methods, null calibration, power |
| Norman et al. 2019, K562 | CRISPRa, single and paired | ~100 singles, ~130 pairs | the same, plus interactions and double-perturbation prediction |

## What is compared

**Guide assignment** (`src/psbench/guides.py`): fixed UMI threshold, threshold
plus minimum guide fraction, top guide only, and a per-guide Gaussian mixture.
Accuracy is measured on simulated guide counts with known truth across a grid
of MOI and signal strength. Public processed data do not ship with ground
truth for guide identity, so real guide-count matrices (if you have one) are
compared by call rate and pairwise agreement instead (`--counts`).

**Perturbation effects** (`src/psbench/effects.py`), each against non-targeting
controls:

- `nb_glm`: negative-binomial GLM on counts, size-factor offset, batch covariate
- `mixed`: linear mixed model with a random intercept per batch
- `welch`: t-test on log-normalised data (reference baseline)
- `mixscape`: pertpy Mixscape, then KO-classified cells versus control

Evaluated by on-target knockdown, DEGs per perturbation, and a **null
calibration**: non-targeting cells are split into pseudo-perturbations and
tested; every call there is a false positive.

**Power** (`src/psbench/power.py`): simulation from control-cell NB parameters
over a grid of perturbed-cell counts and effect sizes, plus empirical
down-sampling of real perturbations.

**Interactions** (`src/psbench/interaction.py`): per-gene
`~ A + B + A:B` model for every Norman double, and the dosage regression
`delta_AB ~ c1 delta_A + c2 delta_B`, with each pair classified as
synergistic, buffering, or additive.

**Prediction** (`src/psbench/model.py`): a small PyTorch gene-embedding network
predicting held-out double-perturbation profiles, against the additive,
no-change, and mean-double baselines, with k-fold cross-validation over doubles.

## Data

Both datasets are loaded through [pertpy](https://pertpy.readthedocs.io), which
downloads and caches them on first use. Nothing is committed to this repository.

```bash
pip install -e ".[dev]"
psbench download --dataset norman2019 --out data/raw/norman2019.h5ad
psbench download --dataset replogle2022_k562_essential --out data/raw/replogle2022_k562_essential.h5ad
```

The Nextflow workflow does this step itself (cached in `data/raw`).

Equivalent one-liners in Python:

```python
import pertpy as pt
norman = pt.data.norman_2019()                       # newer releases: pt.dt.norman_2019()
replogle = pt.data.replogle_2022_k562_essential()    # newer releases: pt.dt.replogle_2022_k562_essential()
```

Original sources, if you prefer to fetch the files yourself:

- Norman et al. 2019, GEO accession [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344)
- Replogle et al. 2022, figshare article 20029387 (search "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq")

**Check the column names once.** pertpy releases differ in how they name the
perturbation, batch, and guide columns and in whether `X` holds raw counts.
After downloading, run

```bash
psbench inspect --dataset norman2019
psbench inspect --dataset replogle2022_k562_essential
```

and edit `config/default.yaml` (`perturbation_col`, `batch_col`, `guide_col`,
`var_symbol_col`) to match. If no raw counts are found, `nb_glm` falls back to
OLS on log-normalised data and a warning is logged.

## Running

Unit tests (synthetic data only, no download):

```bash
pytest -q
```

Workflow smoke test on tiny subsets:

```bash
nextflow run main.nf -profile test,conda
```

Full run:

```bash
nextflow run main.nf -profile conda --n_chunks 16
```

Outputs land in `results/tables/` (CSV per analysis) and `results/report/`
(`summary.md` and figures). Replogle is subsampled to 150 perturbations and
300 cells per perturbation by default; raise `max_perturbations` and
`max_cells_per_perturbation` in `config/default.yaml` for a larger run.

## Layout

```
main.nf, nextflow.config     workflow and profiles
config/default.yaml          datasets, parameters, quick-run overrides
src/psbench/                 data, guides, effects, metrics, power, interaction, model, report, cli
tests/                       pytest suite on a synthetic dataset with planted effects
```

## Notes and limitations

- Guide-assignment accuracy is a simulation result; it shows how the methods
  behave under a stated noise model, not on the real datasets.
- The mixed model is fitted on the target gene plus the 300 most variable
  genes per perturbation for speed, so its FDR is computed over that subset.
- Mixscape is run on single perturbations only.
- The PyTorch model works on pseudo-bulk deltas (one profile per condition),
  so the training set is a few hundred rows; the additive baseline is hard to
  beat at that size and the comparison is reported as is.
