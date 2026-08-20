"""Quick-look aggregation and plots, one module per experiment.

These read the JSON an experiment writes and print/plot it at screen quality;
`ioc.plots` is the separate, paper-quality figure pass over `ioc/data`.

    e1_recovery        recovery vs demonstration count M, and sample efficiency
    e1_noise           recovery vs demonstration noise sigma (E1's headline)
    bench2d_quicklook  2D scaling curves, recovered reward fields, paths
"""
