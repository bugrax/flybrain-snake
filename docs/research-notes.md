# Development decisions

The original exploration considered several bodies for a fly-connectome controller: a goalkeeper, a tethered drone, a browser game, and a Nokia-style Snake simulator. Snake was selected because it made the sensory scene and resulting action easy to inspect without hardware.

The shared technical question was whether a visual-to-descending-neuron subcircuit would preserve lateralized stimulus information. The implementation therefore began with LC10a steering pathways and LC4/LPLC2 escape pathways, then checked propagation and receptive-field tuning before closing the loop around the game.

The main decisions were:

1. Use the real chemical-connectivity graph downstream of a declared visual frontend.
2. Derive receptive-field centers from presynaptic optic-lobe columns, with an approximate geometric transformation.
3. Include adaptation after observing persistent recurrent activity in the isolated subgraph.
4. Reset between discrete decisions to control residual activity.
5. Use graded azimuth matching and leave measured left/right calibration gains unapplied.
6. Keep the historical walled benchmark separate from the public Snake II recreation.
7. Include seed-selection results and full decision telemetry with the video.

Historical performance was modest: mean 2.48 food and best 9 over 50 walled-grid games. The current release demonstrates a selected wraparound run, not an improvement claim against that benchmark. Noise, ablation controls and human comparisons remain possible future experiments. The original private research transcripts are retained locally and excluded from the public repository; these English notes summarize the decisions relevant to the released implementation.
