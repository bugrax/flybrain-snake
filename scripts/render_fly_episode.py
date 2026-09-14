"""Record a real connectome-controlled episode as an English LinkedIn MP4.

Run as a module: uv run python -m scripts.render_fly_episode --seed 6
Playback uses 5 game ticks per second; every tick simulates 250 ms of neural
activity. Frames are real simulation output, with no substituted agent.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
import wave
from pathlib import Path
import numpy as np
from flysnake.nokia import NokiaSnakeGame
from flysnake.controller import FlyController, STEER_TYPES, ESCAPE_TYPES
from flysnake.social import SocialComposer
from flysnake.video import VideoWriter, save_png, find_ffmpeg


def add_audio(silent, output, events, duration):
    """Original synthesized blips: no sampled Nokia audio or music."""
    rate=44100
    samples=np.zeros(int(np.ceil(duration*rate)),np.float32)
    for start,freq,length in events:
        first=int(start*rate); n=min(int(length*rate),len(samples)-first)
        if n <= 0: continue
        t=np.arange(n)/rate
        tone=0.12*np.sin(2*np.pi*freq*t)*np.minimum(t/0.008,1)*np.exp(-t*12)
        samples[first:first+n] += tone
    wav=output.with_suffix('.wav')
    with wave.open(str(wav),'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(samples,-1,1)*32767).astype('<i2').tobytes())
    subprocess.run([find_ffmpeg(),'-v','error','-y','-i',str(silent),'-i',str(wav),
                    '-c:v','copy','-c:a','aac','-b:a','128k','-movflags','+faststart','-shortest',str(output)],check=True)
    silent.unlink(); wav.unlink()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=6)
    ap.add_argument('--out',type=Path,default=Path('media/flybrain-snake-linkedin.mp4'))
    ap.add_argument('--max-steps',type=int,default=500)
    ap.add_argument('--starvation',type=int,default=220)
    ap.add_argument('--level',type=int,choices=range(1,10),default=4)
    ap.add_argument('--maze',type=int,choices=range(6),default=0)
    ap.add_argument('--fps',type=int,default=30)
    ap.add_argument('--frames-per-step',type=int,default=6)
    args=ap.parse_args()
    if min(args.max_steps,args.starvation,args.fps,args.frames_per_step) <= 0:
        ap.error('step limits, fps and frames-per-step must be positive')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    c=FlyController(); net=c.net
    hud=SocialComposer(); game=NokiaSnakeGame(args.seed,args.level,args.maze)
    selected=[]; rng=np.random.default_rng(0)
    for typ,k in [('LC10a',40),('LC4',24),('LPLC2',24)]:
        ids=net.ids(typ); selected.extend(rng.choice(ids,min(k,len(ids)),replace=False).tolist())
    for typ in STEER_TYPES+ESCAPE_TYPES+['DNp01']:
        selected.extend(net.ids(typ).tolist())
    selected=np.array(selected)
    raster=np.zeros((len(selected),1000),bool)
    bars=[('STEER',0,0),('ESCAPE',0,0),('GIANT',0,0)]
    def frame(phase='play',reason=''):
        return hud.frame(game.observe(),raster,bars,phase=phase,reason=reason,
                         neurons=net.n,synapses=int(c.g['n_syn']),seed=args.seed)
    silent=args.out.with_name(args.out.stem+'-silent.mp4')
    telemetry=[]; audio=[(0.3,660,0.15),(0.55,880,0.2)]
    nframes=0; began=time.monotonic(); best=-1
    with VideoWriter(silent,fps=args.fps,size=(1080,1350),preset='fast') as video:
        intro=frame('intro'); save_png(intro,args.out.with_name('cover.png'))
        for _ in range(args.fps*2): video.write(intro); nframes+=1
        for step in range(args.max_steps):
            obs=game.observe(); action=c.act(obs); last=c.last
            spikes=last['spikes']; ms=spikes.shape[1]
            for k in range(args.frames_per_step):
                prev=round(k/args.frames_per_step*ms); upto=round((k+1)/args.frames_per_step*ms)
                new=spikes[selected,prev:upto]
                if new.shape[1]:
                    raster=np.roll(raster,-new.shape[1],axis=1); raster[:,-new.shape[1]:]=new
                counts=spikes[:,:upto].sum(1)
                bars=[]
                for label,types in [('STEER',STEER_TYPES),('ESCAPE',ESCAPE_TYPES),('GIANT',['DNp01'])]:
                    left=sum(int(counts[net.ids(t,'L')].sum()) for t in types)
                    right=sum(int(counts[net.ids(t,'R')].sum()) for t in types)
                    bars.append((label,min(left/160,1),min(right/160,1)))
                rendered=frame(reason=last['reason'])
                video.write(rendered); nframes+=1
            if game.food_eaten > best:
                best=game.food_eaten; save_png(rendered,args.out.with_name('simulation.png'))
            telemetry.append(dict(t=game.t,score=game.score,food_eaten=game.food_eaten,
                                  head=list(game.head),food=list(game.food) if game.food else None,
                                  action=int(action),reason=last['reason'],steer=last['steer'],
                                  escape=last['escape'],giant_fiber=last['gf']))
            game.step(action)
            if game.event in ('food','bonus'):
                audio.append((nframes/args.fps,880 if game.event=='food' else 1320,0.13))
            if step % 50 == 0: print(f'step {step}: points={game.score}, food={game.food_eaten}',flush=True)
            if not game.alive or game.steps_since_food >= args.starvation: break
        outro=frame('outro')
        audio += [(nframes/args.fps,330,0.18),(nframes/args.fps+0.2,220,0.3)]
        for _ in range(args.fps*3): video.write(outro); nframes+=1
    duration=nframes/args.fps
    add_audio(silent,args.out,audio,duration)
    result=dict(seed=args.seed,level=args.level,maze=args.maze,points=game.score,food=game.food_eaten,
                steps=game.t,alive=game.alive,won=game.won,duration_seconds=duration,
                playback_ticks_per_second=args.fps/args.frames_per_step,
                brain_ms_per_tick=c.ms,neuron_count=net.n,synapse_count=int(c.g['n_syn']),
                max_steps=args.max_steps,starvation=args.starvation,
                termination=game.event if not game.alive else ('starvation limit' if game.steps_since_food>=args.starvation else 'step limit'),
                raster_neuron_ids=net.g['body'][selected].tolist(),bar_scale_spikes=160,trace=telemetry)
    args.out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Wrote {args.out}: {duration:.1f}s, {game.food_eaten} food, {game.score} points, {time.monotonic()-began:.1f}s to render')


if __name__ == '__main__': main()
