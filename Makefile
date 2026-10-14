.PHONY: install test download smoke full

install:
	pip install -e ".[dev]"

test:
	pytest -q

download:
	psbench download --dataset norman2019 --out data/raw/norman2019.h5ad
	psbench download --dataset replogle2022_k562_essential --out data/raw/replogle2022_k562_essential.h5ad

smoke:
	nextflow run main.nf -profile test

full:
	nextflow run main.nf -profile conda
