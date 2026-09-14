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
