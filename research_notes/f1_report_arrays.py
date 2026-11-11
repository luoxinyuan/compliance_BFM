from pathlib import Path
import json,hashlib,statistics,datetime
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/f1-l0-array-audit-20261111';R=S/'runtime/research_notes';A=S/'runs/array-audit-03'
load=lambda p:json.loads(p.read_text())
a=load(A/'interventions.json');meta=load(A/'state-config.json');checks=load(A/'checks.json');assert checks['status']=='passed'
names=list(meta[0]['c_pointer_null']); source=[]
for p in sorted((S/'source').glob('*')):
 if p.suffix in ('.c','.h','.cc'):source.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),url='https://github.com/google-deepmind/mujoco/blob/3.3.3/'+('python/mujoco/' if p.name.startswith('structs.') else 'src/engine/')+p.name))
layout=load(S/'prep/pointer_layout.json');lib=P/'.venv/lib/python3.10/site-packages/mujoco/libmujoco.so.3.3.3'
source.append(dict(file='installed libmujoco.so.3.3.3',sha256=hashlib.sha256(lib.read_bytes()).hexdigest()))
report=dict(status='passed_in_current_configuration',checks=checks,states=meta,individual_tests=[{k:v for k,v in x.items() if k!='path'} for x in a],source_identity=source,layout_identity=layout)
(R/'f1_l0_array_audit_results.json').write_text(json.dumps(report,indent=2))

def role(k):
 if k.startswith('efc_AR'):return ('mj_projectConstraint; PGS/noslip的AR稀疏索引','nefc；仅dual且sparse写入，dense不使用这两个索引','core_constraint.c:2025–2031,2185–2201；solver.c:86–94')
 if k in ('dof_island','map_dof2idof','map_idof2dof'):return ('mj_island的自由度映射','nv；map_idof2dof[nidof:nv]是未约束尾部，不应按nidof盲删','island.c:487–545；forward.c:738–768')
 if k in ('efc_island','map_efc2iefc','map_iefc2efc'):return ('mj_island的约束映射','nefc；按island_iefcadr/island_nefc分组','island.c:550–590；forward.c:743–768')
 if k in ('iLD','iM_colind'):return ('mj_island的岛惯性分解/稀疏列索引','容量nC；有效压缩非零数由nidof行的iM_rowadr＋iM_rownnz确定','island.c:538–545；solver.c:934–941')
 if k=='iefc_J':return ('mj_island的分块Jacobian','容量nJ；dense为Σ(island_nefc×island_nv)，sparse为nefc行rowadr＋rownnz有效非零前缀','island.c:592–608；solver.c:960–985')
 if k in ('iefc_JT','iefc_JT_colind'):return ('mj_island的稀疏转置Jacobian','容量nJ；仅sparse路径写入，有效非零数由nidof行JT_rowadr＋JT_rownnz；dense路径不生成','island.c:617–632；solver.c:960–985')
 if k.startswith('iefc_J'):return ('mj_island的稀疏Jacobian索引','nefc；仅sparse路径写入，rowsuper按各island重算','island.c:603–615；solver.c:960–985')
 if k in ('iefc_aref','iefc_force','iefc_state'):return ('mj_fwdConstraint/岛solver的岛参考/力/状态','nefc（按island_iefcadr/island_nefc）；force/aref在求解前gather，state由solver写入','forward.c:729–768；solver.c:946–952')
 return ('mj_island按约束映射gather的solver输入','nefc（按island_iefcadr/island_nefc）','island.c:637–642；solver.c:946–952')
lines=[]
for k in names:
 rr,ll,dep=role(k);tests=[x for x in a if x['arrays']==[k]];assert len(tests)==6 and all(x['all_output_equal'] for x in tests)
 caps='/'.join(str(__import__('math').prod(m['array_shapes'][k])) for m in meta)
 lines.append(f'| `{k}` | {rr} | {ll}；当前C指针NULL，有效长度0 | {caps}（Python形状，不是已分配C容量） | {dep} | 三状态×置零/存档随机，6/6逐值一致；另联合2臂/状态通过 |')
state_lines=[]
for m in meta:
 n=m['counters'];force=sum(v*v for v in m['force'])**.5
 state_lines.append(f"| {m['state_id']} | {m['time']:.3f}s | {force:.6f}N | {n['ncon']} / {n['nefc']} | {n['nisland']} / {n['nidof']} / {n['nA']} | {n['nJ']} |")
text='''# F1 L0逐数组因果审计

## 结论与边界

**已证实：原25项在当前MuJoCo3.3.3、Newton、dense、island禁用、noslip=0配置下都没有已分配的C数组（指针NULL）。不是需要恢复的有效缓存，更不是会随历史改变的隐藏动力学状态。** Python binding以nefc/nv/nJ/nC等形状包装NULL指针，产生NumPy自有、未初始化的临时数组；先前所谓“数组值不同”来自这些临时内存。不得推广为启用island、稀疏/dual求解器或其他配置也可忽略。

156个真正的C指针干预分支＋3条未扰动基线，逐条3秒：25字段逐一置零/存档范围随机、另25字段联合两臂；全部physics/control/q_PD、actor输入/raw、评分与约束计数逐值相同（没有容差）。没有新增训练或科学能力判据变化。

## 真实状态与接触资格

| 状态 | 时间 | 当前伙伴力范数 | ncon / nefc | nisland / nidof / nA | nJ |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(state_lines)+'''

两受力状态均有真实地面约束；第二状态达到10N饱和。无伙伴力状态依然有脚地支撑，不等于空中无碰撞。nisland=0是当前模型没有启用岛路径，不是“没有接触”；不人为开启island来改变已批准物理合同。

## 干预怎样落到引擎，而不是改临时NumPy对象

1. 用当前安装的`mjdata.h`编译项目内只返回`offsetof/sizeof`的小型布局辅助库，读取真实`MjData._address`；没有重编译/替换MuJoCo，也没有改模型/PD。
2. 在每个真实恢复状态，逐项证实25个C指针为NULL。置入独立owned buffer（零值或从原存档同字段有限值池随机抽样），验证Python视图确实与该C缓冲区共享内存；不是修改getattr返回的无后备临时对象。
3. 干预前后完整C结构除指定指针槽之外逐字节hash相同；所有其余已分配数组、积分状态、Task/BFM/力/动作历史及RNG逐值不动。随机填充使用私有generator。
4. 按产生状态的冻结策略继续30个高层决策；standing控制使用原生standing z。每20ms BFM/PD和5ms真实接触均保留。每分支结束25个指针重新全为NULL，全部实际输出与未扰动基线逐值一致。
5. 对照源码：[碰撞阶段](https://github.com/google-deepmind/mujoco/blob/3.3.3/src/engine/engine_collision_driver.c#L271)调用`mj_clearEfc`；[engine_io.h:159](https://github.com/google-deepmind/mujoco/blob/3.3.3/src/engine/engine_io.h#L159)清空arena指针；[forward:176](https://github.com/google-deepmind/mujoco/blob/3.3.3/src/engine/engine_forward.c#L176)仅在enable island时重新构建；[projectConstraint:2029](https://github.com/google-deepmind/mujoco/blob/3.3.3/src/engine/engine_core_constraint.c#L2029)在非dual时直接返回。当前Newton单体求解读取有效efc/M/qLD，而非空的iefc/iM/iLD。
6. Python绑定[structs.cc:775–784](https://github.com/google-deepmind/mujoco/blob/3.3.3/python/mujoco/structs.cc#L775)和[structs.h:1008–1018](https://github.com/google-deepmind/mujoco/blob/3.3.3/python/mujoco/structs.h#L1008)解释了为何NULL仍能呈现非零shape。以上源码与本地header/binary SHA见机器可读结果，不只凭字段名推断。

## 逐数组表

下表的源文件缩写均指官方固定tag3.3.3 `src/engine/engine_*.c`；本地头文件`mjdata.h:410–480`。激活时的规则用于说明依赖，不等于本轮验证了其他求解配置。

| 数组 | 生成者/角色 | 激活时有效长度规则；当前有效域 | 三状态Python可见元素数 | 读写依据 | 干预结果 |
|---|---|---|---|---|---|
'''+ '\n'.join(lines)+'''

## 保存/比较/恢复规则修订（单列）

- **保存**：保留原失败存档及新原始取证文件；新快照追加C指针有效性、模型配置、约束计数。NULL数组的Python值不作为真实状态，规范比较视图为有效长度0。完整原始bytes留作诊断，不制造“这些字节已恢复”的声明。
- **全量逐值比**：integration、全部有C存储的数组（含qacc、warmstart、有效efc_force等）、实际动作/last-z/输入/历史、callback绑定与伙伴状态、RNG及有效约束计数。没有浮点容差或质量门变动。
- **有效域掩码**：根据实际C指针NULL证据置有效长度0；不是按名称猜、不是nefc>0就认为所有iefc可读。当前共有45个NULL数值字段，其中很多本来shape0；旧25非零形状之外还发现iM同属NULL临时数组，必须一并说明，不能留下随机不初始化误报。没有对这第26字段做额外扰动；其无C存储是直接指针证据。
- **恢复**：F1独立入口保留原完整callback/solver/history恢复顺序，只跳过源/目标同时为NULL的假数组复制；源NULL目标非NULL立即断言。所有有效字段继续原精确恢复与物理复验。
- **配置守卫**：绑定3.3.3、35nv/341nC、Newton、dense、enable/disableflags=0、noslip=0；有效nefc/ncon可变化。任一配置或NULL资格改变则不沿用本证明。岛启用/dual等场景必须重新资格。
- **未改**：BFM/PD/10N/K80D8/频率/reward/科学成功定义、旧训练/评估结论、封存集。没有因此宣布L1能力成功。

## 失败与修复历史

`array-audit-01`在非目标字段hash检查停止；`array-audit-02`保存了全量before/after，发现每次读取多个空指针字段都生成不同临时数值，`np.shares_memory`连相同属性的两次读取也不成立。这两次没有合格扰动rollout，不计成功。`array-audit-03`转为上述真实C指针注入，才是156条决定性实验。原a39be58的L0严格失败不追溯改判；本轮证明错误在portable数组有效性定义，补足Issue57 P0-A残余跟踪，不冒充新的FB更新恢复测试。

完整私有轨迹索引`experiments/f1-l0-array-audit-20261111/runs/array-audit-03/`；每分支路径/hash/受扰元素/成本见[f1_l0_array_audit_results.json](f1_l0_array_audit_results.json)。源码见[f1_array_audit.py](f1_array_audit.py)、[f1_pointer_layout.py](f1_pointer_layout.py)、[f1_mjdata_contract.py](f1_mjdata_contract.py)。
'''
(R/'F1_L0_ARRAY_AUDIT.md').write_text(text)
print('array report written',len(a))
