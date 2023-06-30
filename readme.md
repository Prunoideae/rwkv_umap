# RWKV-UMAP: Recording and visualizing transitions of RWKV-LM internal states

Given the fact that the internal state of RWKV is only a `list[Tensor]` and relatively small (like 10MB on disk when dumped with numpy.dump). It is feasible to record states in each step of token generation, and perform UMAP dimensionality reduction to visualize and dig out some hidden information or pattern in it.

The project has 2 files, `record_state_change.py` and `plot_dump.py`. The first one creates an instance of RWKV, and feed it with predefined token inputs iteratively to emulate the process of autoregressive textgen tasks. The second one plots the internal states recorded during the process and plot them out by layer.

**IMPORTANT NOTE:** the pipeline is written to run with `RWKV-world` series models. After some adaptations I think `Raven` can also work but I don't know.

## Usage

1. Install the dependencies for RWKV.
2. Install the dependencies for umap and plotting: `pip install umap-learn colorcet datashader bokeh holoviews scikit-image pynndescent==0.5.8`.
3. Place the model you want to test in `models`.
4. Modify the `record_state_change.py` and run it.
5. Run the `plot_dump.py`, and check output in `plots`.
