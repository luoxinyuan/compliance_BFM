import csv,gzip,json,pickle
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
def signal(trace):
    return np.array([r['force'] for r in trace['physics'] if r['force_time']-trace['start']<2.1-1e-8])
def read(case,name):
    with gzip.open(root/'cue01'/f'{case:02d}-{name}.gz') as f:return pickle.load(f)
def main():
    results=[]
    for case in (4,5,6):
        zero=signal(read(case,'zero')).mean(0)
        lateral={}
        for name,deg in [('right',-30),('front',0),('left',30),('intermediate',15)]:
            x=read(case,name);f=signal(x);mean=f.mean(0);lateral[name]=mean[1]
            tan=np.deg2rad(deg);axis=np.array([np.cos(tan),np.sin(tan)])
            p=f[:,:2]@axis;side=f[:,:2]@np.array([-axis[1],axis[0]])
            corrected=mean-zero
            angle=np.degrees(np.arctan2(corrected[1],corrected[0]))
            nearest=min((-30,0,30),key=lambda d: abs(np.arctan2(np.sin(np.deg2rad(angle-d)),np.cos(np.deg2rad(angle-d)))))
            strict=all(abs(angle-deg)<abs(angle-other) for other in (-30,0,30) if other!=deg) if name!='intermediate' else None
            results.append(dict(case=case,dir=name,hard=x['hard'] is not None,along=p.mean(),cross=side.mean(),vertical=mean[2],vertical_rms=np.mean(f[:,2]**2)**.5,along_rms=np.mean(p*p)**.5,fraction_along_over_abs_vertical=(p>abs(f[:,2])).mean(),background_corrected_deg=angle,nearest=nearest,closest_ok=strict,projection_ok=p.mean()>abs(mean[2])))
        for row in results[-4:]:row['lateral_order']=bool(lateral['left']>lateral['front']>lateral['right'])
        n=read(case,'none');assert n['hard'] is None and np.all(np.array([r['force'] for r in n['physics']])==0)
        z=read(case,'zero');assert z['hard'] is None
    with (root/'cue_axes.csv').open('w') as f:
        w=csv.DictWriter(f,results[0]);w.writeheader();w.writerows(results)
    for r in results:print(r['case'],r['dir'],round(r['background_corrected_deg'],1),r['projection_ok'],r['lateral_order'],r['closest_ok'])
if __name__=='__main__':main()
