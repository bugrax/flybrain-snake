"""Play the Nokia recreation or watch the connectome control it.

    uv run python -m flysnake.play --agent human
    uv run python -m flysnake.play --agent fly --seed 6
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .game import relative_action
from .nokia import NokiaSnakeGame
from .render import NokiaRenderer, load_mono_font


def phone_frame(game, agent, started=True, best=0):
    """Original vector-drawn handset surrounding the 84x48 LCD."""
    img = Image.new('RGB', (660, 940), '#101311')
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((44, 25, 616, 915), 100, fill='#303c4b', outline='#64717c', width=3)
    d.rounded_rectangle((62, 38, 598, 520), 85, fill='#a1a8aa')
    for y in range(66, 99, 9):
        d.rounded_rectangle((268, y, 392, y+3), 2, fill='#4b545b')
    d.text((330, 124), 'NOKIA', font=load_mono_font(28), anchor='mm', fill='#28313a')
    lcd = Image.fromarray(NokiaRenderer(scale=6, bezel=12).render(game.render_frame()))
    img.paste(lcd, (66, 158))
    if not started or game.paused or not game.alive:
        label = 'Snake II' if not started else ('Paused' if game.paused else 'Game over')
        d.rectangle((155, 238, 505, 382), fill='#c4d3a5', outline='#1e2b1a', width=3)
        d.text((330, 272), label, font=load_mono_font(33), anchor='mm', fill='#1e2b1a')
        sub = 'Enter to start' if not started else ('Space to resume' if game.paused else 'R to play again')
        d.text((330, 329), sub, font=load_mono_font(22), anchor='mm', fill='#1e2b1a')
    d.rounded_rectangle((262, 487, 398, 533), 20, fill='#d2d7d8')
    d.text((330, 510), 'MENU', font=load_mono_font(18), anchor='mm', fill='#27333c')
    keys = [('1',''),('2','UP'),('3',''),('4','LEFT'),('5','PAUSE'),('6','RIGHT'),('7',''),('8','DOWN'),('9','')]
    for i,(number,label) in enumerate(keys):
        x, y = 125 + (i % 3)*145, 579 + (i // 3)*77
        d.rounded_rectangle((x-44,y-22,x+79,y+34), 23, fill='#bfc7ca', outline='#e7eded', width=1)
        d.text((x-13,y+3), number, font=load_mono_font(25), fill='#24333b', anchor='mm')
        d.text((x+29,y+5), label, font=load_mono_font(10), fill='#35434a', anchor='mm')
    d.text((330, 828), f'{agent.upper()}  /  LEVEL {game.level}  /  MAZE {game.maze}', font=load_mono_font(17), anchor='mm', fill='#dce6de')
    d.text((330, 859), f'BEST {best:05d}  /  FLYBRAIN SNAKE', font=load_mono_font(16), anchor='mm', fill='#9cafae')
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--agent', choices=('human','fly'), default='human')
    ap.add_argument('--level', type=int, choices=range(1,10), default=4)
    ap.add_argument('--maze', type=int, choices=range(6), default=0)
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--smoke-test', action='store_true', help='Render three frames headlessly and exit')
    args = ap.parse_args()
    if args.smoke_test:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import pygame
    pygame.mixer.pre_init(22050, -16, 1, 512)
    pygame.init()
    screen = pygame.display.set_mode((660, 940))
    pygame.display.set_caption('Flybrain Snake / Nokia Snake II')
    clock = pygame.time.Clock()
    game = NokiaSnakeGame(args.seed, args.level, args.maze)
    ctrl = None
    if args.agent == 'fly':
        from .controller import FlyController
        ctrl = FlyController()
    save = Path.home() / '.flybrain-snake' / 'best.txt'
    try:
        best = int(save.read_text())
    except (OSError, ValueError):
        best = 0
    def beep(freq, seconds=0.07):
        if pygame.mixer.get_init():
            t = np.arange(int(22050 * seconds)) / 22050
            samples = (np.sin(2*np.pi*freq*t) * np.exp(-t*25)*6000).astype(np.int16)
            pygame.sndarray.make_sound(samples).play()
    mapping = {pygame.K_UP:(0,-1),pygame.K_2:(0,-1),pygame.K_w:(0,-1),
               pygame.K_DOWN:(0,1),pygame.K_8:(0,1),pygame.K_s:(0,1),
               pygame.K_LEFT:(-1,0),pygame.K_4:(-1,0),pygame.K_a:(-1,0),
               pygame.K_RIGHT:(1,0),pygame.K_6:(1,0),pygame.K_d:(1,0)}
    started = args.agent == 'fly' or args.smoke_test
    pending = deque(maxlen=2)
    running, elapsed, frames = True, 0, 0
    while running:
        elapsed += clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_ESCAPE:
                    running = False
                elif key == pygame.K_RETURN:
                    started, elapsed = True, 0
                elif key in (pygame.K_SPACE, pygame.K_p, pygame.K_5):
                    game.toggle_pause(); elapsed = 0
                elif key == pygame.K_r:
                    game.reset(args.seed); pending.clear(); started = True; elapsed = 0
                    if ctrl: ctrl.prev_obs = None
                elif key in (pygame.K_l, pygame.K_m):
                    level = game.level % 9 + 1 if key == pygame.K_l else game.level
                    maze = (game.maze + 1) % 6 if key == pygame.K_m else game.maze
                    game = NokiaSnakeGame(args.seed, level, maze)
                    started = False; pending.clear(); elapsed = 0
                    if ctrl: ctrl.prev_obs = None
                elif key in mapping and len(pending) < 2 and started and game.alive and not game.paused:
                    base = pending[-1] if pending else game.heading
                    desired = mapping[key]
                    if desired != base and relative_action(base, desired) is not None:
                        pending.append(desired)
        if started and game.alive and not game.paused and elapsed >= game.tick_ms:
            elapsed = 0
            if ctrl:
                action = ctrl.act(game.observe())
            else:
                desired = pending.popleft() if pending else game.heading
                action = relative_action(game.heading, desired)
                action = action if action is not None else 0
            game.step(action)
            if game.event in ('food','bonus'): beep(880 if game.event == 'food' else 1320)
            if not game.alive: beep(150, 0.3)
            best = max(best, game.score)
        frame = phone_frame(game, args.agent, started, best)
        screen.blit(pygame.image.frombuffer(frame.tobytes(), (660,940), 'RGB'), (0,0))
        pygame.display.flip(); frames += 1
        if args.smoke_test and frames >= 3: running = False
    if not args.smoke_test:
        save.parent.mkdir(exist_ok=True)
        save.write_text(str(best))
    pygame.quit()


if __name__ == '__main__':
    main()
