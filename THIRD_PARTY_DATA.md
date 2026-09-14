# MaleCNS data attribution

`data/subgraph_snake.npz` is a derived subset of **MaleCNS v1.0**, released by the Male CNS Connectome Project. The dataset is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Attribution: Male CNS Connectome Project; FlyEM / HHMI Janelia Research Campus, University of Cambridge Department of Zoology, MRC Laboratory of Molecular Biology, and Google Research.

- [Project and associated publications](https://male-cns.janelia.org/)
- [Source downloads and license statement](https://male-cns.janelia.org/download/)
- [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

## Source files

Downloaded from the `v1.0/connectome-data/flat-connectome/` directory of the project's `flyem-male-cns` Google Cloud Storage bucket:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

The full source tables are not included in Git. `scripts/download_data.py` downloads them from the official public bucket.

## Changes made

`flysnake/brain.py` selects visual neurons, intermediary partners and descending/readout neurons. It retains internal connections with at least five synapses, maps predicted neurotransmitters to signed weights, multiplies connection counts by 0.275 mV, and calculates receptive-field centers from presynaptic column annotations. The cache contains source body IDs, cell types, sides, transmitter predictions, signed CSR weights, receptive-field centers and aggregate counts.

These transformations produce 3,189 neurons, 111,905 connections and 2,100,007 source synapses. Neural dynamics and the visual/motor interfaces are original code and should not be attributed as measurements made by the dataset authors. The dataset authors do not endorse this project.
