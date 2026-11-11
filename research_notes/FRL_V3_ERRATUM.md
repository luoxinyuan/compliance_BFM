# Issue59实现错误隔离

## 2026-11-11：归档路径拼接

- runtime561af1d，counterfactual-01，120.3865job秒，失败在第一个归档路径由Path加str的表达式；模型初始化/第一个源prefix已执行，但未保存任何反事实结果，0学习更新。
- 源码/输出/日志/sidecar保持；修复用Path分段连接，5项CPU风险单位/边界/path检查通过。
- ac1d51a的counterfactual-02为独立目录、独立运行；不覆盖旧失败，不改变协议、reward、概率/物理边界。
- P1回报评分另核对原终止absorbing tail：原字段c.reward_integral不含该尾项，最终审计单独保留optimized-score-audit，出现受影响分支时必须以此纠正并记录影响，不覆盖原输出。当前尚未据此作科学结论。

旧Issue58及其erratum不改。

## 2026-11-11：完成采集后的汇总序列化

- ac1d51a/counterfactual-02采完32状态280分支后，在historical-credit bool_写入失败；原job/traceback保持failed，1019.1553秒计账。
- 不重跑物理；从逐分支完整JSON恢复8条credit、32状态汇总和存在性gate，complete-recovery记录outcomes hash和全文件资格。
- 32条原连续轨迹逐值核验通过；新CPU审计才建立可用数据资格，不把原进程改判成功。
- 0/280分支终止，score helper未显式扣absorbing尾项的问题未影响本轮数字；新helper明确保留该尾项，并加回归测试。
- 6项CPU事件单位/路径/连续边界/终止尾项检查通过；原PPO/GAE/保持/相对门源码逐字节或AST相同。
