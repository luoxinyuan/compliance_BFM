# Issue53：三条关键真实物理对照

全部为真实 MuJoCo qpos 重放，25fps、正常1×；显示真实力、接触点、anchor、heading与root轨迹。包含完整接触与释放恢复，仅省略0–3秒无接触warmup。01/02有固定斜视与俯视，03为四格固定斜视。

| 视频 | 在比较什么 |
|---|---|
| 01-exact-prefix-teacher-recovery.mp4 | 三行都是同一个冻结S4000。上：自己执行；中：Teacher只替换4.7–4.8s；下：Teacher接管4.7–6.7s，其余全由S4000执行。展示早期进入状态对后续闭环的影响，不是三个新模型。 |
| 02-corrected-postcontact-recovery.mp4 | 8cm短请求，上为S4000；下为修复完整状态恢复后的Teacher接管5.4–6.8s，之后交回S4000。局部窗口减力，但后续保持仍有失败，未剪掉。 |
| 03-valid-student-new-short-and-long.mp4 | 左S4000、右同一个旧数据BC500；上为新6cm请求，下为新59cm持续引导。BC500在看这些留出条件前固定，只增加旧数据监督，没有新交互标签或PPO。 |

S4000 SHA256：8f5e7287e974deedba1a8d841dd65d7692e944e278bd4a091f1ec224f167a479。
BC500 SHA256：a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e。
BFM/原PD、K80D8/10N保持不变。新Student仅有理想仿真力＋本体/动作历史权限，不等于G1真机传感已实现。

释放后屏幕标为non-task geometry的anchor距离不是主动跟随误差。视频02的接管窗内改善不代表交回后整个回合都好。视频03的长距保持牵拉、偏航和释放后多余运动仍需改进，不能只看走起来。

原跨reset快照缺陷已修复；受其影响的混合交互BC不作为成功展示。来源、完整失败与原始模型见[Issue53](https://github.com/GaoE05/robot-research-bridge/issues/53)。三份MP4均已逐文件SHA核对、时长/帧率检查及完整解码；其余审计放诊断目录，不放本视频文件夹。
