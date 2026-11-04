"""Publish compact diagnostics from saved M39-B development analysis only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m39b-retention-20261104')
O = Path(__file__).with_name('m39b_artifacts')

def main():
    O.mkdir(exist_ok=True)
    src = R / 'development-closeout'
    learn = json.loads((src/'learning.json').read_text())
    physical = json.loads((src/'development-physics.json').read_text())
    results = json.loads((src/'development-results.json').read_text())
    for name, obj in [('development-results', results), ('final-config', json.loads((src/'final-config.json').read_text()))]:
        (O/(name+'.json')).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    compact = {arm: {k:v for k,v in data.items() if k!='batches'} for arm,data in learn.items()}
    (O/'training-complete.json').write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n')
    compact_physics = {arm:{key:{k:v for k,v in data.items() if k!='quarter_second_curve'}
                            for key,data in tasks.items()} for arm,tasks in physical.items()}
    (O/'development-phases.json').write_text(json.dumps(compact_physics, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for name, color in [('raw','tab:blue'), ('action','tab:orange')]:
        rows=learn[name]['batches']; dev=learn[name]['development']
        axes[0,0].plot([d['iteration'] for d in dev],[d['score'] for d in dev],'-o',label=name,color=color)
        axes[0,1].plot([d['iteration'] for d in rows],[d['retention_diagnostics']['qpd_rms_deg'] for d in rows],label=name,color=color)
        axes[1,0].plot([d['iteration'] for d in rows],[100*d['optimization']['gradient_components'][0]['ratio'] for d in rows],label=name,color=color)
        axes[1,1].plot([d['iteration'] for d in rows],[d['force_rms'] for d in rows],label=name,color=color)
    titles=['16-task development cost (lower better; initial=1.5)', 'Saved-state retention q_PD RMS (deg; not physics)',
            'Weighted retention / PPO gradient norm (%)', 'Stochastic training force RMS (task mix; not dev score)']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.set_xlabel('PPO batch');ax.grid(alpha=.2);ax.legend()
    fig.savefig(O/'development-curves.png',dpi=150);plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    for col,key in enumerate(['sequence','sequence_right']):
        for name,color in [('initial','black'),('raw','tab:blue'),('action','tab:orange')]:
            curve=physical[name][key]['quarter_second_curve'];t=[r['time'] for r in curve]
            x0=curve[0]['root_xyz'][0]
            axes[0,col].plot(t,[100*(r['root_xyz'][0]-x0) for r in curve],label=name,color=color)
            axes[1,col].plot(t,[r['torso_deg'] for r in curve],label=name,color=color)
            axes[2,col].plot(t,[sum(sum(v*v for v in f)**.5 for f in r['point_force_xyz']) for r in curve],label=name,color=color)
        axes[1,col].plot(t,[r['target_deg'] for r in curve],'k--',label='partner yaw')
        axes[0,col].set_title(key+' (development; real physics)')
        spec=physical['initial'][key]['condition']
        for row in range(3):
            ax=axes[row,col]
            for x in [spec['onset'],spec['rotation_onset'],spec['rotation_return'],spec['release']]:ax.axvline(x,alpha=.2,color='gray')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    for row,label in enumerate(['Root X displacement (cm)','Torso yaw (deg)','Sum point force norms (N)']): axes[row,0].set_ylabel(label)
    for ax in axes[-1]:ax.set_xlabel('Simulation time (s), evaluation only')
    fig.savefig(O/'development-transitions.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
