# CURRENT_STATE — 新会话先读

最后核对：2026-11-05 CST。**Issue44（Issue43后续：负迁移与有效恢复）完成并停止；没有PPO/Student或下一阶段排队。**

入口：[结果](PRIOR_RECOVERY_RESULTS.md) · [聚合证据](recovery_artifacts/physical-metrics.json) · [Issue44](https://github.com/GaoE05/robot-research-bridge/issues/44) · [Master18](https://github.com/GaoE05/robot-research-bridge/issues/18)。长期纲领RESEARCH_OVERVIEW不变；旧Library2026-10-28仅历史快照。

## 科学目标与保留基线

Contact-conditioned whole-body behavior selection/modulation/transition：按实际接触及身体状态组织局部手臂、躯干/root与必要支撑/行走，不写死先臂后脚。冻结BFM/原PD。I/T30/S0/D2及M0–M43历史保留，有限直行Teacher→ideal-force Student与双向小转腰不被本次失败否定；不宣称真机已有精确力传感。

- BFM SHA33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
- I SHA72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a；T30 c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33。
- S0 0ef3d43b03803cbf226949bd5272517dede6a4b76c069c4794f900cebc4ead12；D2 9e4630896f9849c7ae1f68fa0058d6e23e1792095ef2a362dfd30c17baccf7f5。
- 旧B多行为初始化e1112cd0b20d1d8462ac3efbf183cb1289b5679612b37a286232e4a5eaedf00c。Issue43两臂PPO best均iteration0，没有PPO修复结论。原A/B和首/次轮都保留。

## Issue44实际结果

1. 22回合I/A/B与接管，16/16完整前缀hash一致；I/A历史qpos/qvel/force/anchor逐值复现。B在接触前4.0s已出现持续z/q_PD偏离，真实q随后偏离、cap饱和在后，不能仅归因力太小。
2. B→I4.3/5.0/5.5/6s可以恢复80cm起步；7s接管误差23.87cm、力8.74N，未作为优质标签。I→B到5.5s后可继续走，说明行走条件能力尚存，早期反馈/历史演化被破坏。8cm中B→I反而变差，不能全面覆盖B。
3. 原750数据无完全相同输入冲突。原B参数变化约A101倍；B上局部/行走梯度方向冲突是机制线索，不是不可学习证明。
4. C0复现旧B1500，actor逐值差0。C1加入3个有效接管起点恢复短段＋有限可靠启动域五子步q_PD保持。浮点边界遗漏首条标签后，从保存决策准确补全30条；两组共同追加500，总4000，固定2000最终版本。旧1500与缺陷保留，不按终检重选。
5. 最终80cm C0/C1误差40.395/6.059cm，恢复起步；但8cm为2.872/4.936cm，C1左足路径20.72cm，C0仅.33cm，局部优势丢失。新7cm重复该取舍（2.589/4.703cm；足路径.30/21.80cm）。新60cm C1 7.008cm仍不及I5.429cm，且有额外偏航。新arm-ready18cm未超过I。5N长距C1 13.543cm，I6.883cm。
6. 106完整新物理回合无安全终止，但记录3个短暂双足support=false tick（精确条件见报告）。q_PD峰66.46°与真实5ms关节峰3.06°分开。释放后不考核旧anchor跟随。

**有效恢复成立；同一模型局部/长距行为共存仍不成立。没有启动条件性PPO，不替换I/T30/S0，不拼C0/C1。**

## 当前瓶颈与下一决策（未执行）

当前全参数共享MLP的混合监督＋软动作保持能救回walking，却牺牲局部响应。现有证据不要求增力、重训BFM或随机搜z，也不能唯一归因输入不可辨识/网络容量不足。

唯一建议：下一阶段做结构性保旧对照——同I/数据/预算，当前全参数MLP对比冻结I主干＋零初始化可学习条件latent adapter，保留合法z和观测权限，不增加手工force→skill或固定动作路径。须由用户授权；不是已证明修法。当前不继续1500更新循环，不启动PPO/Student、force-history六组或真机。

## 身份、成本与接力

最终C0 SHA ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c；C1 8ead3da1a2844fa8ac4cae3543df7aae93021202a8d8335af2f164fd2076455d。两者为correct-C0/C1/*.pt，2000累计更新，runtime f7559db0706fb19331aa153b2db7ea7b854be159。个人gaoe05/research最终文档commit见Issue44；不改main。

run `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/prior-recovery-20261105`；物理/训练/终检/失败/精确SHA全存。记录job-wall约2732秒，另119秒CPU聚合与短检查；约1.8GiB，4000监督/0RL/0Student，最多GPU2/5，owned进程已退出。

Mac `~/Projects/compliance-bfm-diagnostics/videos/prior-recovery/`：01因果接管；02 C0/C1最终长距；03新7cm。仅3条1×真实qpos MP4+INDEX，SHA/全解码/画面核验通过，不把视频生成当行为成功。

接力先读本文件、长期纲领、PRIOR_RECOVERY_RESULTS和Issue44；按新授权再核验资源，不能复用旧PID或空闲快照。共享6023c6e checkout保持；不动BFM/PD/权重/main/SoftSONIC/他人任务。M11初轮及continuation早停历史仍独立保留，不能改写成等预算终检。
