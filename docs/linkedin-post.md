I gave a fruit fly's wiring diagram the controls to Nokia Snake II.

3,189 simulated neurons.
2.1 million source synapses.
Zero game-policy training episodes.

The attached video is a real run: it finds food, turns, and eventually crashes into itself. Four food items. One very nostalgic game over.

I built Flybrain Snake using a visual-to-motor subcircuit from Janelia's MaleCNS connectome. Visual input drives simulated neurons, their spikes propagate through the recorded connections, and a readout rule converts left/right motor activity into turns.

The scope matters: this is a simulated subcircuit with a modeled visual frontend, tuned parameters, and an explicit motor readout. The source wiring is biological; the game interface is engineered.

My favorite part is watching the motor spikes change just before the snake turns.

Code, playable game, assumptions, and the full decision trace:
https://github.com/bugrax/flybrain-snake

What game would you give this tiny nervous system next?

#Neuroscience #Connectomics #Python #OpenSource #SnakeII
