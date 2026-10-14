nextflow.enable.dsl = 2

params.config         = "${projectDir}/config/default.yaml"
params.datasets       = 'norman2019,replogle2022_k562_essential'
params.combo_datasets = 'norman2019'
params.outdir         = 'results'
params.cache_dir      = 'data/raw'
params.n_chunks       = 8
params.quick          = false
params.quick_flag     = params.quick ? '--quick' : ''

process DOWNLOAD {
    tag "$ds"
    storeDir params.cache_dir

    input:
    val ds
    path cfg, stageAs: 'CFG'

    output:
    tuple val(ds), path("${ds}.h5ad")

    script:
    """
    psbench --config CFG ${params.quick_flag} download --dataset ${ds} --out ${ds}.h5ad
    """

    stub:
    "touch ${ds}.h5ad"
}

process PREPARE {
    tag "$ds"

    input:
    tuple val(ds), path(raw)
    path cfg, stageAs: 'CFG'

    output:
    tuple val(ds), path("${ds}.prepared.h5ad")

    script:
    """
    psbench --config CFG ${params.quick_flag} prepare --dataset ${ds} --input ${raw} --out ${ds}.prepared.h5ad
    """

    stub:
    "touch ${ds}.prepared.h5ad"
}

process GUIDES {
    publishDir "${params.outdir}/tables", mode: 'copy'

    input:
    path cfg, stageAs: 'CFG'

    output:
    path "guide_benchmark.csv"

    script:
    """
    psbench --config CFG ${params.quick_flag} guides --out guide_benchmark.csv
    """

    stub:
    "touch guide_benchmark.csv"
}

process EFFECTS {
    tag "${ds}:${method}:${chunk}"
    publishDir "${params.outdir}/tables", mode: 'copy'

    input:
    tuple val(ds), path(prepared), val(method), val(chunk)
    path cfg, stageAs: 'CFG'

    output:
    path "effects_${ds}_${method}_chunk${chunk}.csv"

    script:
    """
    psbench --config CFG ${params.quick_flag} effects --input ${prepared} --method ${method} \\
        --chunk ${chunk} --n-chunks ${params.n_chunks} \\
        --out effects_${ds}_${method}_chunk${chunk}.csv
    """

    stub:
    "touch effects_${ds}_${method}_chunk${chunk}.csv"
}

process MIXSCAPE {
    tag "$ds"
    publishDir "${params.outdir}/tables", mode: 'copy'

    input:
    tuple val(ds), path(prepared)
    path cfg, stageAs: 'CFG'

    output:
    path "effects_${ds}_mixscape_chunk0.csv"
    path "mixscape_${ds}_summary.csv"

    script:
    """
    psbench --config CFG ${params.quick_flag} mixscape --input ${prepared} \\
        --out effects_${ds}_mixscape_chunk0.csv \\
        --summary-out mixscape_${ds}_summary.csv
    """

    stub:
    """
    touch effects_${ds}_mixscape_chunk0.csv mixscape_${ds}_summary.csv
    """
}

process NULL_CALIBRATION {
    tag "${ds}:${method}"
    publishDir "${params.outdir}/tables", mode: 'copy'

    input:
    tuple val(ds), path(prepared), val(method)
    path cfg, stageAs: 'CFG'

    output:
    path "null_${ds}_${method}.csv"

    script:
    """
    psbench --config CFG ${params.quick_flag} null --input ${prepared} --method ${method} --out null_${ds}_${method}.csv
    """

    stub:
    "touch null_${ds}_${method}.csv"
}

process POWER {
    tag "$ds"
    publishDir "${params.outdir}/tables", mode: 'copy'

    input:
    tuple val(ds), path(prepared)
    path cfg, stageAs: 'CFG'

    output:
    path "power_${ds}*.csv"

    script:
    """
    psbench --config CFG ${params.quick_flag} power --input ${prepared} --out power_${ds}.csv
    """

    stub:
    "touch power_${ds}.csv power_${ds}_downsample.csv"
}

process INTERACTION {
    tag "$ds"
    publishDir "${params.outdir}/tables", mode: 'copy'

    input:
    tuple val(ds), path(prepared)
    path cfg, stageAs: 'CFG'

    output:
    path "interaction_${ds}_*.csv"

    script:
    """
    psbench --config CFG ${params.quick_flag} interaction --input ${prepared} \\
        --out interaction_${ds}_summary.csv \\
        --genes-out interaction_${ds}_genes.csv
    """

    stub:
    "touch interaction_${ds}_summary.csv interaction_${ds}_genes.csv"
}

process MODEL {
    tag "$ds"
    publishDir "${params.outdir}/tables", mode: 'copy'

    input:
    tuple val(ds), path(prepared)
    path cfg, stageAs: 'CFG'

    output:
    path "model_${ds}_cv.csv"

    script:
    """
    psbench --config CFG ${params.quick_flag} model --input ${prepared} --out model_${ds}_cv.csv
    """

    stub:
    "touch model_${ds}_cv.csv"
}

process REPORT {
    publishDir "${params.outdir}/report", mode: 'copy'

    input:
    path tables
    path cfg, stageAs: 'CFG'

    output:
    path "report/*"

    script:
    """
    psbench --config CFG ${params.quick_flag} report --results-dir . --out report
    """

    stub:
    """
    mkdir report && touch report/summary.md
    """
}

workflow {
    cfg = file(params.config)

    datasets = Channel.fromList(params.datasets.tokenize(','))
    combo_ds = params.combo_datasets.tokenize(',')

    prepared = PREPARE(DOWNLOAD(datasets, cfg), cfg)

    chunks    = Channel.fromList((0..<params.n_chunks).toList())
    per_pert  = Channel.of('nb_glm', 'mixed', 'welch')
    null_meth = Channel.of('nb_glm', 'mixed', 'welch')

    guides   = GUIDES(cfg)
    effects  = EFFECTS(prepared.combine(per_pert).combine(chunks), cfg)
    mixscape = MIXSCAPE(prepared, cfg)
    nulls    = NULL_CALIBRATION(prepared.combine(null_meth), cfg)
    powers   = POWER(prepared, cfg)

    combos   = prepared.filter { ds, h5 -> ds in combo_ds }
    inter    = INTERACTION(combos, cfg)
    models   = MODEL(combos, cfg)

    all_tables = guides
        .mix(effects, mixscape[0], mixscape[1], nulls, powers, inter, models)
        .flatten()
        .collect()

    REPORT(all_tables, cfg)
}
