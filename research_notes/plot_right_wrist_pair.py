"""Plot already recorded data only. No model or simulator imports."""
import argparse,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output
    b=json.loads((root/'no_force.json').read_text());f=json.loads((root/'tether.json').read_text())
    bp,fp=b['physics'],f['physics'];t=np.array([r['time'] for r in fp]);force=np.array([r['force'] for r in fp])
    fig,axs=plt.subplots(4,1,figsize=(10,11),sharex=True,layout='constrained')
    axs[0].plot(t,force[:,0],label='Applied Fx');axs[0].plot(t,np.linalg.norm(force,axis=1),label='Applied |F|')
    axs[0].axhline(5,color='gray',ls=':',label='5 N cap');axs[0].set_ylabel('Actual force (N)');axs[0].legend(loc='upper right')
    dx=np.array([r['wrist'] for r in fp])-np.array([r['wrist'] for r in bp])
    dr=np.array([r['wrist_relative_root'] for r in fp])-np.array([r['wrist_relative_root'] for r in bp])
    root_delta=np.array([r['root'] for r in fp])-np.array([r['root'] for r in bp])
    axs[1].plot(t,dx[:,0]*1000,label='Wrist world X');axs[1].plot(t,dr[:,0]*1000,label='Wrist relative-root X')
    axs[1].plot(t,root_delta[:,0]*1000,label='Root world X');axs[1].set_ylabel('Tether - no-force (mm)');axs[1].legend()
    axs[2].plot(t,[r['tilt_deg'] for r in bp],label='No-force tilt');axs[2].plot(t,[r['tilt_deg'] for r in fp],label='Tether tilt')
    axs[2].set_ylabel('Root tilt (deg)');axs[2].legend(loc='upper left')
    ax2=axs[2].twinx();ax2.plot(t,[r['root'][2] for r in bp],color='gray',ls='--',label='No-force height');ax2.plot(t,[r['root'][2] for r in fp],color='purple',ls=':',label='Tether height')
    ax2.set_ylabel('Root height (m)');ax2.legend(loc='upper right')
    for episode,label,ls in [(bp,'No-force','--'),(fp,'Tether','-')]:
        for i,side in enumerate(['L','R']):axs[3].plot(t,[r['foot_normal_forces'][i] for r in episode],ls=ls,label=label+' '+side)
    axs[3].set_ylabel('Foot contact normal (N)');axs[3].legend(ncol=2);axs[3].set_xlabel('Simulation time (s)')
    for ax in axs:
        ax.axvspan(1,3,color='orange',alpha=.09);ax.axvspan(3,4.5,color='green',alpha=.06)
        ax.axvline(4.5,color='black',ls=':',alpha=.4);ax.grid(alpha=.2)
    fig.suptitle('Original frozen BFM: 3 cm +X right-wrist tether / matched no-force\nSeed 0; K=80 N/m, D=8 Ns/m; onset 1 s, ramp to 3 s, release 4.5 s')
    fig.savefig(root/'right_wrist_pair.png',dpi=160);fig.savefig(root/'right_wrist_pair.pdf');plt.close(fig)

if __name__=='__main__':main()
