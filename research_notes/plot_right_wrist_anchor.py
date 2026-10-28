"""Plot saved simulations only; never interpolate or pad a terminated rollout."""
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
    data={n:json.loads((out/(n+'.json')).read_text()) for n in ['no_force','closed_loop','replay']}
    result=json.loads((out/'result.json').read_text());origin=result['anchor_source']['origin']
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained');ax=axes.ravel()
    colors={'no_force':'gray','closed_loop':'tab:blue','replay':'tab:orange'}
    labels={'no_force':'No force','closed_loop':'Live BFM + tether','replay':'Action replay + tether'}
    for name,e in data.items():
        rows=e['physics'];t=np.array([r['time'] for r in rows]);color=colors[name]
        force=np.array([r['force'] for r in rows]);wrist=np.array([r['wrist'] for r in rows]);root=np.array([r['root'] for r in rows])
        if name!='no_force':
            for i,c in enumerate(['tab:red','tab:green','tab:blue']):ax[0].plot(t,force[:,i],color=c,ls='-' if name=='closed_loop' else '--',label=name+' F'+'XYZ'[i])
        ax[1].plot(t,(wrist[:,0]-origin[0])*1000,color=color,label=labels[name])
        if name=='closed_loop':ax[1].plot(t,[(r['anchor'][0]-origin[0])*1000 for r in rows],color='black',ls=':',label='Anchor X')
        b=data['no_force']['physics'][:len(rows)];broot=np.array([r['root'] for r in b]);bwrist=np.array([r['wrist'] for r in b])
        if name!='no_force':
            ax[2].plot(t,((wrist-root)-(bwrist-broot))[:,0]*1000,color=color,label=labels[name])
        ax[3].plot(t,[r['tilt_deg'] for r in rows],color=color,label=labels[name])
        ax[4].plot(t,root[:,2],color=color,label=labels[name])
        normal=np.array([r['foot_normal_forces'] for r in rows])
        if name!='no_force':
            ax[5].plot(t,normal[:,0],color=color,label=name+' left')
            ax[5].plot(t,normal[:,1],color=color,ls='--',label=name+' right')
        if e['metrics']['fall_rate']:
            for a in ax:a.axvline(t[-1],color=color,ls=':',alpha=.65)
    titles=['Actual force components (N)','Wrist world X from common anchor origin (mm)',
            'Wrist-root X difference from no-force (mm)','Root tilt (deg)','Root height (m)','MuJoCo foot normal force (N)']
    for a,title in zip(ax,titles):
        a.set_title(title);a.set_xlabel('Simulation time (s)');a.grid(alpha=.2)
        a.axvspan(1,3,color='orange',alpha=.06);a.axvline(4.5,color='black',ls=':',alpha=.35)
        a.set_xlim(0,6);a.legend(fontsize=8)
    fig.suptitle('Activation-aligned anchor: original closed-loop BFM vs no-force action replay\nSame Tether/PD; K80 D8 cap5 N; +3 cm, seed0; traces stop at actual termination')
    fig.savefig(out/'anchor_replay.png',dpi=160);fig.savefig(out/'anchor_replay.svg');plt.close(fig)

if __name__=='__main__':main()
