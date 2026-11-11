# Issue57 P0-A portable数组残余跟踪

2026-11-11回填，证据来自Issue60 L0，见[F1_L0_ARRAY_AUDIT](F1_L0_ARRAY_AUDIT.md)、[完整续验](F1_PREFLIGHT.md)及[F1_ERRATUM](F1_ERRATUM.md)。

当前MuJoCo3.3.3、Newton/dense/island禁用、35nv配置下，原25个不稳定数组的实际C指针均NULL；另发现iM及shape0的同类NULL字段。Python以非零逻辑形状包装NULL而产生未初始化临时内存，不能算有效动力学状态。

已补齐原缺失的有效域证明：本地header/对应tag引擎分支＋真实C指针读取；两伙伴受力/一无伙伴力状态，逐数组置零/存档范围随机和联合干预，156条3秒真实续跑逐值一致。未改其他有效C状态，且按字节重读复验；27完整状态恢复与后继重复通过。

修复限F1个人独立入口：NULL→NULL不复制临时数组；保存有效域元数据；其他积分/有效数组/solver/history/RNG继续精确恢复。其他求解器、island启用、不同物理模型必须重新核实，不能全局忽略island或AR字段。

**本次关闭的是当前配置下portable数组有效范围的证据缺口，不是重新执行FB更新恢复试验。** 没有FB训练/agent.update，没有重写D2旧失败或宣称新的compiled恢复资格；P0-A梯度累加顺序与后续数值验证继续引用各自原报告。
