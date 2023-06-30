import os
import umap
import umap.plot
import numpy as np
from rwkv_contrib.debug_tools import StateDump
from rwkv_contrib.tokenizer import RWKVTokenizer
import matplotlib.pyplot as plt
from os import cpu_count

PROMPT_FILE = "test_prompt.txt"
DUMP_PREFIX = "states/state"
LAYER_COUNT = 32
LINE_CLUSTER_COUNT = 5
# use half of the cores as per hammer bundling requires 2 cores per process
# be extra cautious if you're on Windows, as multiprocessing works weird
PROCESS_COUNT = cpu_count() // 2
N_NEIGHBORS = 100
MIN_DIST = 0.25

# load the dump
dump = StateDump(DUMP_PREFIX)
layers, labels = dump.loads()

tokenizer = RWKVTokenizer()
input_tokens = tokenizer.encode(open(PROMPT_FILE).read())
line_labels = []
line_index = 0
for input_token in input_tokens:
    line_lower = (line_index // LINE_CLUSTER_COUNT) * LINE_CLUSTER_COUNT
    line_upper = line_lower + LINE_CLUSTER_COUNT - 1
    # align the line labels to the count of layers
    line_labels.extend([f"Newline {line_lower:02}-{line_upper:02}"] * LAYER_COUNT)
    if "\n" in tokenizer.decode([input_token]):
        line_index += 1


def plot_layer(layer_index: int):
    """
    Performs UMAP, connectivity, and PCA on the layer.
    """
    global layers, labels, line_labels

    sliced_layer = []
    sliced_labels = []
    sliced_lines = []

    for label, layer, line in zip(labels, layers, line_labels):
        # label format: {token_count}-l-{layer_index}
        # select only the layer data we need
        if label.endswith(f"-l-{layer_index}"):
            sliced_layer.append(layer)
            sliced_labels.append(label)
            sliced_lines.append(line)

    # convert everything we need into ndarray as UMAP requires it
    sliced_labels = np.array(sliced_labels)
    sliced_layer = np.array(sliced_layer)
    stage_tokens = [int(l.split("-")[0]) for l in sliced_labels]
    stage_tokens = np.array(stage_tokens)
    sliced_lines = np.array(sliced_lines)

    os.makedirs(f"plots/layer{layer_index}", exist_ok=True)

    # fit data into UMAP, then plot, only one mapper here so all plots will have a same layout
    mapper = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST).fit(sliced_layer)

    # plot the UMAP, the color is based on at which token the layer is recorded
    # blue means the layer is recorded at the beginning of the prompt (0)
    # red means the layer is recorded at the end of the prompt (-1)
    axes = umap.plot.points(mapper, values=stage_tokens, cmap="jet")
    axes.set_title(f"Layer {layer_index}")
    plt.savefig(f"plots/layer{layer_index}/umap.png")
    plt.close()

    # plot the UMAP, the label is based on which line the layer is recorded
    axes = umap.plot.points(mapper, labels=sliced_lines, theme="fire")
    axes.set_title(f"Layer {layer_index}")
    plt.savefig(f"plots/layer{layer_index}/umap2.png")
    plt.close()

    # plot the connectivity of the UMAP
    # the label is based on which line the layer is recorded, and so do the color
    # however, most of the time the color might not be visible due to the large amount of points
    axes = umap.plot.connectivity(mapper, labels=sliced_lines, show_points=True, edge_bundling="hammer")
    axes.set_title(f"Layer {layer_index}")
    plt.savefig(f"plots/layer{layer_index}/connectivity.png")
    plt.close()

    # plot the PCA of the UMAP
    # Principle Component Analysis is a method to reduce the dimension of the data
    # by combining it with the UMAP, we can see some more global structure of the 
    # data while still keeping the local structure by UMAP
    # it's useful when you need to test if the umap result is good by checking if PCA shows
    # a globally agreed trend.
    axes = umap.plot.diagnostic(mapper, diagnostic_type="pca")
    axes.set_title(f"Layer {layer_index}")
    plt.savefig(f"plots/layer{layer_index}/pca.png")
    plt.close()


from multiprocessing import Pool

if __name__ == "__main__":
    with Pool(PROCESS_COUNT) as pool:
        # check if we're on windows and need to freeze support
        if os.name == "nt":
            from multiprocessing import freeze_support

            freeze_support()

        pool.map(plot_layer, range(LAYER_COUNT))
