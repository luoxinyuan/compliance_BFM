# CURRENT_STATE — F-RL-v2结项；三项档案诊断完成

最后核对2026-11-11。[Issue58](https://github.com/GaoE05/robot-research-bridge/issues/58) ·
[执行结果](ISSUE58_EXECUTION_RESULTS.md) · [锁定协议](FRL_V2_FB_REPAIR_PROTOCOL.md)。

## 当前状态

原有界GPU作业已停止；GPU空闲状态须实时核对，不把旧快照当授权。没有自动续训、Student、正式FB1M或5M。
A/B各128×768=98304真实on-policy高层决策均已完成，合计196608。
每臂489640个50Hz控制、1958560物理tick，998完整回合；无物理边界终止。
A/B均按原开发保护选iteration0，封存按选定初始化执行，不能当新模型成功。
原BC500维持基线；本报告不是全面否认PPO已有的有限开发收益。

## 最新授权诊断已完成（2026-11-11）

[一页摘要](FRL_V2_DIAGNOSTICS_SUMMARY.md) · [三项报告](FRL_V2_ARCHIVE_DIAGNOSTICS.md)。纯CPU现有档案复算，0训练/新物理/Teacher查询/封存读取，4job小时内完成并停止。
- 探索链抽查B64/B128/A128：干净推进代理59/39/19→正PPO优势16/21/11→raw均值方向保留11/9/3（合法z10/11/3）。只有观测代理，不把负优势自动判critic错；没有确切dev状态的采样物理反事实，真实因果三分比例未识别。
- BC500现行相对自门40/40通过，但mid/stop滑移峰值p50/max=.395/.848与.765/1.103m/s；完整尾部分布/CDF已列，不改门。
- reward独立物理重算24条/11156控制步差异0；质量等级32/32、Pareto21/21排序一致；打滑成功高于干净成功0/3且只有1父组，不能泛化成reward无问题。
- 原B128未晋级、BC500基线不变；暴露诊断不反选模型。旧结项与erratum保留，不自动新训。

## 科学结果

- A局部改善，但中距及质量退化；B显著降低接触前函数/实际前缀漂移，mid/stop有真实改善。
- B64开发示例：near9.00cm/6.48N；mid7.25cm/5.72N；stop/continue8.93cm/6.22N。
  初始化对应9.03/6.45、7.89/6.21、11.62/7.60；均为active窗口3D误差/真实力RMS。
  B64仍有slip/yaw尾部未过，不是正式晋级模型，不按封存反选。
- 末批mean D_pre A13.5116、B.01109；32个开发main父组接触前root差中位数A.10244cm/B.000509cm。
  函数保持有效降低漂移，但不足以保证共同质量收益。单seed方法对照，无唯一根因结论。
- 概率/GAE/物理合同未放宽：LP最大.00012207<.0003，接受jointKL<.02。

## 加速实际结果

快速记录独立真实评价59.72→46.63s，1.281×；完整档逐值相同。
中央canonical推理1/2/4 worker热采样60.39/41.01/35.82s，1/1.473/1.686×，完整8档与原进程相同。
worker冷启动104/206/412s，不能把热采样倍数当总job/训练FPS；相对快速单进程，4 worker约1.30×。
BFM批量shadow有约4e-7动作差，未执行；当前正式训练仍原num_envs=1，尚无并行PPO更新/恢复资格。

## FB工程边界

P0-A eager固定16更新逐值通过；额外真实physics/control/RNG相同，但portable最终state有23个island辅助数组差异。
严格门仍未关闭，不删除字段或降容差改判。compiled固定16测试出现second-backward错误，原失败保留。
P0-B三源30父组/90配对，原4个关节边界降到0，但滑移/偏航退化，资格失败；工程8192采样和更新未启动。
不自动重跑正式FB。

## 身份、预算与接力

BC500 a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e；
BFM33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361。
正式A/B19b3117；FB工程d757bee；记录profile92e4a60；并行prototype edade56。
本轮sidecar实测F11.008h/P0-A.508h/P0-B.412h，准备reserve另列；原240h未重置。
证据stage experiments/frl-v2-fb-repair-20261110，实际models/SHA/分项数据见执行报告和私有账本。
发布HEAD不是运行源码；全部旧结项、负例、erratum和基线保留；不改main/共享资产/他人任务。

本次探索—优势—保留、质量门与reward档案诊断已完成，见本页最新摘要；没有自动新实验。
剩余因果缺口不能靠本轮观测代理补成结论；FB的portable数组/compiled与动作质量仍保持旧边界。
新会话先读本页、OVERVIEW、三项诊断报告及Issue58/Master18，再核实际进程；不要重复启动。

执行错误的影响隔离见[Issue58 erratum](ISSUE58_EXECUTION_ERRATUM.md)，原始失败不覆盖。
