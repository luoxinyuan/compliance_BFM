import csv, collections, statistics, pathlib
root=pathlib.Path(__file__).resolve().parent
def read(name):
    with (root/name).open() as f:return list(csv.DictReader(f))
rows=read('pairs.csv')
scores=read('scores.csv')
metrics=('old','lag','long','both')
print('# 辅助目标与信用时域的固定四格复算\n')
print('仅读取已暴露实际轨迹。四列分别为原辅助回报、增加连接位置成本、延长折扣、同时采用两项。折扣由每20ms的0.98变为0.995。位置项归一尺度10cm。各格按相同物理窗口起算，比较同父同分支；冻结模型没有更新。')
print('\n|轨迹对照|窗口|可判对数 / 祖先数|原|加位置|长折扣|两项|\n|---|---|---:|---:|---:|---:|---:|')
for family,labels in [('运动源',['weak','full']),('恢复站立',['early','late']),('安静站立',['U'])]:
 for label in labels:
  for stage in ['onset','midmove','hold']:
   rr=[r for r in rows if r['parent']!='50' and r['contrast']==label and r['stage']==stage and r['old']]
   stats=[sum(float(r[k]) for r in rr) for k in metrics]
   print('|',family+' '+label,'|',stage,'|',f'{len(rr)} / {len(set(r["ancestor"] for r in rr))}','|',' | '.join(f'{x:g}' for x in stats),'|')
print('\n表为排对的对数（平局算半）。祖先与兄弟分支相关，同一祖先不会因重新进入静站而获得独立身份。详细分组、未入分母与质量资格见 pairs.csv。')
print('\n## #50 发现例，从 onset 到释放\n')
for r in scores:
 if r['parent']=='50' and r['stage']=='onset':print('-',r['version'],', '.join(f'{float(r[k]):.2f}' for k in metrics))
print('\n## 质量与完整性\n')
for stage in ['onset','midmove','hold']:
 rr=[r for r in rows if r['qualified']=='True' and r['stage']==stage and r['parent']!='50']
 print('-',stage,'非发现质量合格可判对数',len(rr),'祖先',sorted(set(r['ancestor'] for r in rr)))
rr=[r for r in scores if r['stage']=='onset']
print('- 轨迹',len(rr),'；归档条件组',len(set(r['parent'] for r in rr)),'；非发现运动祖先身份',sorted(set(r['ancestor'] for r in rr if r['parent'] not in ['50','quiet'])))
assert all(abs(float(r['old'])-float(r['lag']))<1e-8 and abs(float(r['long'])-float(r['both']))<1e-8 for r in scores if r['complete']=='True' and (r['branch']=='no_force' or r['stage']=='release'))
print('- 无力与释放后的完整窗口逐轨迹核对，两种时域下新增位置成本均为零。')
