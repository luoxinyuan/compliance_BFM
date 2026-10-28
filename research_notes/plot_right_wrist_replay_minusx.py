"""Plot existing traces only, with original world signs and no padded failures."""
import argparse,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    out=parser.parse_args().output
    data={n:json.loads((out/(n+'.json')).read_text()) for n in ['no_force','no_force_replay','minus_x','plus_x_reference']}
    fig,axes=plt.subplots(4,3,figsize=(15,13),layout='constrained')
    for name,color,label in [('plus_x_reference','tab:blue','+X archived'),('minus_x','tab:orange','-X new')]:
        rows=data[name]['physics'];base=data['no_force']['physics'][:len(rows)];t=np.array([r['time'] for r in rows])
        for i in range(3):
            axes[0,i].plot(t,[r['force'][i] for r in rows],color=color,label=label)
            axes[1,i].plot(t,[(r['wrist_relative_root'][i]-b['wrist_relative_root'][i])*1000 for r,b in zip(rows,base)],color=color,label=label)
            axes[2,i].plot(t,[(r['root'][i]-b['root'][i])*1000 for r,b in zip(rows,base)],color=color,label=label)
        axes[3,0].plot(t,[r['root'][2] for r in rows],color=color,label=label)
        axes[3,1].plot(t,[r['tilt_deg'] for r in rows],color=color,label=label)
        for i,style in [(0,'-'),(1,'--')]:axes[3,2].plot(t,[r['foot_normal_forces'][i] for r in rows],color=color,ls=style,label=label+(' left' if i==0 else ' right'))
    for name,style in [('no_force','-'),('no_force_replay','--')]:
        rows=data[name]['physics'];t=[r['time'] for r in rows]
        axes[3,0].plot(t,[r['root'][2] for r in rows],color='gray',ls=style,label=name)
        axes[3,1].plot(t,[r['tilt_deg'] for r in rows],color='gray',ls=style,label=name)
    for i,axis in enumerate('XYZ'):
        axes[0,i].set_title('Actual F'+axis+' (N)')
        axes[1,i].set_title('Wrist-root '+axis+' delta vs no force (mm)')
        axes[2,i].set_title('Root '+axis+' delta vs no force (mm)')
    for i,title in enumerate(['Root height (m)','Root tilt (deg)','Foot normal force (N)']):axes[3,i].set_title(title)
    for ax in axes.ravel():
        ax.set_xlim(0,6);ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        ax.axvspan(1,3,color='orange',alpha=.06);ax.axvline(4.5,color='black',ls=':',alpha=.3)
    fig.suptitle('Original frozen BFM: full no-force replay control and matched +/-X anchor\nK80 D8 cap5 N; seed0; common no-force t=1s origin; no retraining')
    fig.savefig(out/'replay_minusx.png',dpi=150);fig.savefig(out/'replay_minusx.svg');plt.close(fig)

if __name__=='__main__':main()
