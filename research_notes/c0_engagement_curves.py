"""Plot fixed-condition development curves; mixed sampled force is not a task curve."""
import argparse,json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args();x=json.loads(a.data.read_text())
    fig,ax=plt.subplots(2,3,figsize=(13,7.5));colors={'4701':'#1768ac','4702':'#d46b15'};csvrows=[]
    for seed,r in x['training'].items():
        dev=sorted(r['development'].items());ts=[int(k.split('-')[1]) for k,_ in dev];col=colors[seed]
        ax[0,0].plot(ts,[v['score']['score'] for _,v in dev],'o-',color=col,label=seed)
        for g,style in [('near','-'),('far','--')]:ax[0,1].plot(ts,[v['score']['selection']['groups'][g] for _,v in dev],'o'+style,color=col,label=seed+' '+g)
        for label,style in [('local7','-'),('local9','--')]:
            rows=[next(z for z in v['episodes'] if z['label'].split('__')[0]==label) for _,v in dev]
            ax[0,2].plot(ts,[z['error_cm'] for z in rows],'o'+style,color=col,label=seed+' '+label)
        for label,style in [('long65','-'),('long80','--')]:
            rows=[next(z for z in v['episodes'] if z['label'].split('__')[0]==label) for _,v in dev]
            ax[1,0].plot(ts,[z['error_cm'] for z in rows],'o'+style,color=col,label=seed+' '+label)
        its=[z['iteration'] for z in r['curves']]
        ax[1,1].plot(its,[z['optimization']['epochs'][-1]['joint_gaussian_KL'] for z in r['curves']],'.-',color=col,label=seed)
        ax[1,2].plot(its,[z['optimization']['raw_mean_change_rms'] for z in r['curves']],'.-',color=col,label=seed)
        for k,v in dev:
            for e in v['episodes']:csvrows.append(dict(seed=seed,iteration=int(k.split('-')[1]),condition=e['label'],error_cm=e['error_cm'],force_rms_N=e['force_rms'],cap_fraction=e['cap_fraction'],boundary=e['boundary']))
    for a1,title,y in [(ax[0,0],'Fixed development score','cost / original C0'),(ax[0,1],'Near vs far trade-off','cost / original C0'),(ax[0,2],'Near requests (same conditions)','3D error RMS (cm)'),(ax[1,0],'Far requests (same conditions)','3D error RMS (cm)'),(ax[1,1],'Accepted PPO joint KL','KL'),(ax[1,2],'Actual mean command update','raw mean change RMS')]:
        a1.set_title(title);a1.set_ylabel(y);a1.set_xlabel('PPO batch (768 training decisions each)');a1.grid(alpha=.25);a1.legend(fontsize=7)
    ax[0,0].axhline(1,color='grey',lw=.8);ax[0,1].axhline(1,color='grey',lw=.8)
    fig.suptitle('C0-start PPO: both development-selected at batch 8; no heldout selection',fontsize=13)
    fig.tight_layout();fig.savefig(a.output,dpi=160);plt.close(fig)
    with a.output.with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
if __name__=='__main__':main()
