# Snake II recreation

The public player uses `NokiaSnakeGame`. The earlier `SnakeGame` remains available for the historical experiment and its existing baseline agents.

The recreation provides a native 84 × 48 monochrome framebuffer, an 8-pixel score strip, a 21 × 10 playfield, open-edge wrapping, irreversible direction changes, a connected segmented snake, a feeding animation, two-cell timed critters, levels 1–9, open play and five mazes, pause/restart, and synthesized beeps.

Ordinary food grows the snake by one cell and awards the selected level in points. A bonus appears after every fifth ordinary food item when room is available, lasts 40 ticks, grows the snake by one cell, and awards `level × remaining ticks` points. Levels use tick intervals of 300, 260, 220, 180, 150, 120, 100, 80 and 60 ms. Offline video playback can intentionally use a different tick rate.

Maze 0 has no walls; maze 1 is a closed border; mazes 2–5 contain original internal layouts. Body and wall collisions end the game. Moving into the departing tail is legal when not growing. Filling the accessible board ends in a win. Food and bonus placements exclude the body and walls. Seeded placement is deterministic.

## Fidelity boundary

This is an independent implementation inspired by the 3310 experience. It is not a Nokia ROM emulator and has not been validated against original hardware. The five layouts are original approximations; exact timing, scoring, sprite shapes, bonus frequency and sounds are not claimed to match the firmware. No Nokia ROM, copied sprite sheet, or original audio is distributed.

The gameplay references confirm wraparound, bonus creatures, maze selection and directional keypad controls. These features were missing from the original project and are now implemented.

## References

- [Nokia / Microsoft Devices Blog: Snake 2k](https://blogs.windows.com/devices/2012/06/25/snake-2k-slithers-onto-your-nokia-lumia/) describes directional 2/4/6/8 controls, temporary critters and a 3310-style handset.
- [Snake '97 developer's Snake 2 guide](https://snake97.com/guides/snake-2/) describes open borders, bonus creatures and maze choices.
- [Snake 2000 developer's reconstruction notes](https://2nplusone.github.io/Snake2000/) describe four-pixel cells, two-cell critters, directional segments and the feeding animation.
