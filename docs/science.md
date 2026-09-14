# What is biological, and what is modeled?

Flybrain Snake simulates a **subcircuit**, not the complete fly brain. Its connectivity comes from MaleCNS v1.0. Neuron dynamics, the conversion of a game scene into visual drive, and the conversion of neural activity into actions are explicit models.

| Layer | Implementation | Provenance |
| --- | --- | --- |
| Neurons and connectivity | 3,189 neurons, 111,905 directed connections, 2,100,007 synapses | Derived from MaleCNS v1.0 chemical connectivity |
| Subgraph | LC10a, LC11, LC4, LPLC2, LC6, LC16 seeds; first-hop partners receiving at least 20 seed synapses; second-hop descending neurons; designated readout types; retain internal edges with at least 5 synapses | Selection is ours; retained connections and counts are from the dataset |
| Transmitter signs | ACh, DA, OA, 5-HT positive; GABA, glutamate, histamine negative; 23 unknown assignments treated as excitatory | Transmitter predictions are from MaleCNS; mapping to uniform signs is a simplifying model, especially for neuromodulators |
| Neuron dynamics | LIF: rest/reset −52 mV, threshold −45 mV, membrane time constant 20 ms, refractory period 2.2 ms, synaptic time constant 5 ms, delay 1.8 ms, 0.275 mV per synapse | Parameters from Shiu et al. (2024); this implementation has a 1 ms time step and rounds the delay to 2 steps |
| Adaptation | Add 1 mV per spike, decay time constant 150 ms | Our addition; absent from the referenced parameter set |
| Receptive fields | Synapse-weighted centers of each LC neuron's presynaptic columns, using `assignedOlHex1/2` | Centers derived from the connectome; conversion from hex coordinates to viewing angles is approximate |
| Visual frontend | Small object → LC10a; nearby approaching edges → LC4 / LPLC2; egocentric camera on the snake's head | Explicit game-state model, not a simulated retina |
| Decisions | 250 ms of freshly reset network activity per game tick | Our discretization; no neural memory across decisions |
| Motor readout | Bilateral steering, escape and giant-fiber populations converted to relative turns | Our deterministic rule using real neuron populations |
| Missing connections | Electrical synapses / gap junctions | Not supplied by the chemical-connectivity table; not inserted by hand |

## Visual input

The eyes turn with the snake's heading. Food is represented as a small dark object. Walls and body segments ahead are represented as approaching edges. Each visual neuron receives drive according to a Gaussian match between the stimulus azimuth and its receptive-field center, with azimuth width **25 degrees**. Elevation matching remains broad because the game is two-dimensional.

Retinotopy coverage in the extracted graph: LC4 **126/126**, LPLC2 **185/185**, LC10a **248/275**. The model does not propagate pixels through retina, lamina and medulla. The displayed food position is used directly, including in wraparound play. The model neither optimizes a toroidal shortest path to food nor pursues bonus creatures as a distinct stimulus. Obstacle sensing respects open edges and maze walls.

## From activity to turns

Steering uses `DNa02 + DNa01 + DNa03 + DNa13 + DNa15`. A sufficiently strong asymmetric response turns toward the more active side: minimum population count 6, with an absolute asymmetry of at least 3 spikes and 40% of the total.

Escape uses `DNp02 + DNp03 + DNp04 + DNp11`. At a count of at least 8, a lateralized response triggers a turn away from the more active side; the asymmetry threshold is 30%. When both sides respond, steering activity or the quieter escape side breaks the tie. `DNp01` provides an emergency trigger when a looming stimulus is present.

Left/right calibration responses are measured, but gains remain **1.0 / 1.0**. Early calibration attempts were removed because the graded frontend already gave approximately symmetric frontal responses. The thresholds, gains and frontend settings were chosen during development. Zero training episodes means there is no learned game policy; it does not mean the complete system is parameter free.

## Why adaptation and resets?

The isolated recurrent subcircuit continued firing after input ended under the original implementation without adaptation. Spike-frequency adaptation reduced this persistent activity. Resetting between decisions prevents residual activation from dominating the next discrete game step. Both are engineering interventions and limit claims about biological fidelity. Missing inhibitory context is a plausible explanation for reverberation, rather than an experimentally established cause here.

The model uses the chemical graph. Electrical components of LC4 → giant fiber and giant fiber → TTMn are absent. Consequently, this is not a complete reconstruction of the fly's escape reflex.

## Propagation evidence

Development measurements with sustained unilateral input produced:

- Left LC10a → DNa02: left 70 / right 4 spikes; DNa13: 112 / 0.
- Left LC4 → DNp04: 60 / 0; DNp02: 52 / 10; DNp11: 53 / 6; giant fiber: 71 / 69.
- Direct synapse counts: LC4 → DNp04 11,597; LC4 → DNp01 6,362; LPLC2 → DNp01 4,862; LC4 → DNp02 4,209; LC4 → DNp11 3,666. These selected direct pathways were ipsilateral in the extracted data.

Steering-population azimuth sweep, left/right spikes: −60° 29/0; −30° 101/0; −15° 77/1; **0° 14/15**; +15° 1/43; +30° 0/87; +60° 0/44. Escape sweep: −60° 98/18; 0° 126/126; +60° 0/106.

Reproduce using `uv run python -m scripts.gono_go_propagation` and `uv run python -m scripts.tuning_sweep`. These results demonstrate propagation within this model; they are not recordings from a living fly.

## Reading the video

The spike raster displays a fixed sample of LC neurons followed by steering, escape and giant-fiber neurons. Its window spans 1,000 simulation milliseconds, concatenated across independently reset decisions. The exact body IDs are in the episode JSON. Motor bars sum the relevant populations on each side and saturate at 160 spikes. The scene stays visible while the corresponding decision's neural activity is progressively revealed.

The release video uses seed 10, level 4, open maze, 250 ms of neural time per step, and 8 frames per step at 30 fps. Playback is therefore 3.75 game ticks per second. Seed selection and complete decision telemetry are included. The recorded run collected four ordinary food items and ended after 99 steps; it did not collect a bonus.

## Sources

- [MaleCNS project, release notes, and downloads](https://male-cns.janelia.org/).
- [Shiu et al., A Drosophila computational brain model reveals sensorimotor processing, Nature 634, 210–219 (2024)](https://doi.org/10.1038/s41586-024-07763-9).
- [Reference implementation by the paper's authors](https://github.com/philshiu/Drosophila_brain_model).
