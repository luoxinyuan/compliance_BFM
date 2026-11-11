# Issue59实现错误隔离

## 2026-11-11：归档路径拼接

- runtime561af1d，counterfactual-01，120.3865job秒，失败在第一个归档路径由Path加str的表达式；模型初始化/第一个源prefix已执行，但未保存任何反事实结果，0学习更新。
- 源码/输出/日志/sidecar保持；修复用Path分段连接，5项CPU风险单位/边界/path检查通过。
- ac1d51a的counterfactual-02为独立目录、独立运行；不覆盖旧失败，不改变协议、reward、概率/物理边界。
- P1回报评分另核对原终止absorbing tail：原字段c.reward_integral不含该尾项，最终审计单独保留optimized-score-audit，出现受影响分支时必须以此纠正并记录影响，不覆盖原输出。当前尚未据此作科学结论。

旧Issue58及其erratum不改。
