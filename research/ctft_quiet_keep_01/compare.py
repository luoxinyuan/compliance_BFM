from pathlib import Path
import json
root=Path(__file__).resolve().parent
old=root.parent/'ctft_quiet_01/resumed'
read=lambda p:json.loads(p.read_text())
summary={}
for label,path,actor in [('control',old,'U'),('keep',root/'evaluation','keep')]:
    x={b:read(path/f'{actor}-{b}.json') for b in ['continue','no_force','stop']}
    summary[label]=dict(unloaded_cm=x['no_force']['root_path']*100,unloaded_yaw_deg=x['no_force']['yaw_abs_deg'],increment_cm=100*(x['continue']['root_forward']-x['no_force']['root_forward']),arms={})
    for b,r in x.items():
        summary[label]['arms'][b]={k:r[k] for k in ['lag_rms','force_rms','root_forward','slip_result','complete_release','release_root_path','release_yaw','hard','alternating_support_changes','stopped']}
        if b=='no_force':assert r['force_rms']==0
a,b=summary['control'],summary['keep']
summary['directional_gate']=dict(path=b['unloaded_cm']<a['unloaded_cm'],yaw=b['unloaded_yaw_deg']<a['unloaded_yaw_deg'],progress=b['increment_cm']>=.9*a['increment_cm'],force=b['arms']['continue']['force_rms']<=1.1*a['arms']['continue']['force_rms'],lag=b['arms']['continue']['lag_rms']<=1.1*a['arms']['continue']['lag_rms'],hard=all(not r['hard'] for r in b['arms'].values()))
(root/'comparison.json').write_text(json.dumps(summary,indent=2))
