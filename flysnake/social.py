"""English feed-video layout. All activity panels receive real simulation data."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from .render import HudComposer, NokiaRenderer, load_mono_font

BG = '#111611'
FG = '#f0f3e6'
LIME = '#d4f58b'
MUTED = '#96a38e'


class SocialComposer(HudComposer):
    def __init__(self):
        super().__init__(width=1080, height=1350, screen_scale=10, raster_window=1000)
        self.nokia = NokiaRenderer(scale=10, bezel=14)
        self.raster_box = (54, 860, 650, 1070)
        self.bars_box = (668, 860, 1026, 1070)

    def frame(self, obs, raster, bars, *, phase='play', reason='', neurons=3189, synapses=2100007, seed=6):
        img = Image.new('RGB', (1080,1350), BG)
        d = ImageDraw.Draw(img)
        def text(x,y,s,size=24,color=FG,anchor='la'):
            d.text((x,y),s,font=load_mono_font(size),fill=color,anchor=anchor)
        d.rounded_rectangle((54,42,74,62), 5, fill=LIME)
        text(90,40,'FLYBRAIN SNAKE',22,LIME)
        text(1026,42,'BY BUGRAX',19,MUTED,'ra')
        if phase == 'outro':
            text(52,102,'Tiny brain.',72)
            text(52,184,'Big nostalgia.',72,LIME)
        else:
            text(52,102,f'{neurons:,} neurons.',72)
            text(52,184,'One Nokia game.',72,LIME)
        text(54,285,'A simulated fruit-fly subcircuit takes the controls.',25,MUTED)
        # The handset surrounds, but never resamples, the 84 x 48 framebuffer.
        d.rounded_rectangle((54,338,1026,848),32,fill='#303c46',outline='#526353',width=2)
        text(96,358,'NOKIA',21,'#c9d2d5')
        text(983,358,'SNAKE II',19,'#c9d2d5','ra')
        # Fit the exact LCD at scale 9; keep enough space for the bezel and brand.
        screen = Image.fromarray(NokiaRenderer(scale=9,bezel=12).render(obs['frame']))
        img.paste(screen,((1080-screen.width)//2,389))
        self._draw_raster(img,d,raster)
        # Compact bilateral bars with explicit labels and a shared scale.
        d.rounded_rectangle(self.bars_box,10,fill='#161d17',outline='#303a30')
        text(690,873,'MOTOR SPIKES',19,MUTED)
        text(708,909,'LEFT',15,'#78beff')
        text(998,909,'RIGHT',15,'#ffaa5a','ra')
        for i,(label,left,right) in enumerate(bars[:3]):
            y=947+i*36
            text(839,y,label,15,FG,'mm')
            d.line((693,y,793,y),fill='#334333',width=9)
            d.line((885,y,1005,y),fill='#334333',width=9)
            if left: d.line((793-int(100*min(left,1)),y,793,y),fill='#78beff',width=9)
            if right: d.line((885,y,885+int(120*min(right,1)),y),fill='#ffaa5a',width=9)
        for x,value,label in [(54,f'{obs["score"]:05d}','GAME POINTS'),(385,f'{synapses/1e6:.1f}M','SOURCE SYNAPSES'),(742,'0','TRAINING EPISODES')]:
            text(x,1094,value,43,LIME)
            text(x,1149,label,17,MUTED)
        if phase == 'intro':
            caption='Real wiring. Simulated spikes. Retro reflexes.'
        elif phase == 'outro':
            verdict='BOARD COMPLETE' if obs.get('won') else ('GAME OVER' if not obs['alive'] else 'RECORDING LIMIT')
            caption=f'{verdict} / {obs.get("food_eaten",0)} food / seed {seed}'
        else:
            caption=reason.upper()
        d.line((54,1201,1026,1201), fill='#354130',width=1)
        text(540,1238,caption,24,FG,'mm')
        text(540,1290,'github.com/bugrax/flybrain-snake',23,LIME,'mm')
        text(540,1325,'MaleCNS v1.0 / modeled vision + readout / no learning',16,MUTED,'mm')
        return np.asarray(img)


class LandscapeComposer(HudComposer):
    """1920x1080 composition with a complete LCD and a separate telemetry column."""

    def __init__(self):
        super().__init__(width=1920, height=1080, screen_scale=12, raster_window=1000)
        self.nokia = NokiaRenderer(scale=12, bezel=14)
        self.raster_box = (1280, 462, 1856, 682)
        self.bars_box = (1280, 706, 1856, 944)

    def frame(self, obs, raster, bars, *, phase='play', reason='', neurons=3189, synapses=2100007, seed=6):
        img = Image.new('RGB', (self.width, self.height), BG)
        d = ImageDraw.Draw(img)

        def text(x, y, value, size=24, color=FG, anchor='la'):
            d.text((x, y), value, font=load_mono_font(size), fill=color, anchor=anchor)

        d.rounded_rectangle((64, 45, 85, 66), 5, fill=LIME)
        text(101, 42, 'FLYBRAIN SNAKE', 25, LIME)
        text(1856, 45, 'BY BUGRAX', 22, MUTED, 'ra')
        headline = 'Tiny brain. Big nostalgia.' if phase == 'outro' else f'{neurons:,} neurons. One Nokia game.'
        text(61, 100, headline, 70, LIME)
        text(64, 190, 'A simulated fruit-fly subcircuit takes the controls.', 29, MUTED)

        # Preserve every source pixel at integer scale, with generous outer margins.
        d.rounded_rectangle((64, 266, 1248, 954), 32, fill='#303c46', outline='#526353', width=2)
        text(101, 287, 'NOKIA', 24, '#c9d2d5')
        text(1210, 287, 'SNAKE II / 84 x 48', 22, '#c9d2d5', 'ra')
        screen = Image.fromarray(self.nokia.render(obs['frame']))
        img.paste(screen, (64 + (1184 - screen.width) // 2, 332))

        d.rounded_rectangle((1280, 266, 1856, 438), 12, fill='#161d17', outline='#303a30')
        for x, value, lines in [
            (1305, f'{obs["score"]:05d}', ('GAME', 'POINTS')),
            (1497, f'{synapses / 1e6:.1f}M', ('SOURCE', 'SYNAPSES')),
            (1695, '0', ('TRAINING', 'EPISODES')),
        ]:
            text(x, 290, value, 43, LIME)
            for row, label in enumerate(lines):
                text(x, 363 + row * 26, label, 19, MUTED)

        self._draw_raster(img, d, raster, label='SPIKE RASTER / last 1 s')
        d.rounded_rectangle(self.bars_box, 12, fill='#161d17', outline='#303a30')
        text(1304, 722, 'MOTOR SPIKES', 24, MUTED)
        text(1305, 766, 'LEFT', 18, '#78beff')
        text(1830, 766, 'RIGHT', 18, '#ffaa5a', 'ra')
        for i, (label, left, right) in enumerate(bars[:3]):
            y = 818 + i * 45
            text(1568, y, label, 20, FG, 'mm')
            d.line((1305, y, 1490, y), fill='#334333', width=12)
            d.line((1646, y, 1831, y), fill='#334333', width=12)
            if left:
                d.line((1490 - int(185 * np.clip(left, 0, 1)), y, 1490, y), fill='#78beff', width=12)
            if right:
                d.line((1646, y, 1646 + int(185 * np.clip(right, 0, 1)), y), fill='#ffaa5a', width=12)

        if phase == 'intro':
            caption = 'Real wiring. Simulated spikes. Retro reflexes.'
        elif phase == 'outro':
            verdict = 'BOARD COMPLETE' if obs.get('won') else ('GAME OVER' if not obs['alive'] else 'RECORDING LIMIT')
            caption = f'{verdict} / {obs.get("food_eaten", 0)} food / seed {seed}'
        else:
            caption = reason.upper()
        text(64, 989, caption, 25)
        text(1856, 989, 'github.com/bugrax/flybrain-snake', 25, LIME, 'ra')
        text(64, 1042, 'MaleCNS v1.0 / modeled vision + readout / no learning', 19, MUTED)
        text(1856, 1042, 'REAL SIMULATION / 16:9', 19, MUTED, 'ra')
        return np.asarray(img)
