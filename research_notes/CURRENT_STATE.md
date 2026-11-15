# Current — Issue66 E/I v2：源实现已修复，P0核心合同通过，进入P1冻结物理资格

2026-11-15。Yi整体批准v2与独立新增6,000秒，并追加授权“这种你自己的实现问题可以自己修正后继续”。源重复的原失败/erratum保留；修复后12个arm-ready真实前缀及66对共同窗口无重复，原父组来源与实际执行祖先分开记录。原BFM/PD/normalizer不动。

[P0进展与全部失败修复](CONTACT_GOAL_EI_V2_P0_PROGRESS.md) · [机器证据/成本](contact_goal_ei_v2_p0_progress.json) · [不可变执行锁](ei_v2_locks/protocol.json) · [观测schema](ei_v2_locks/contact_schema.json)。初始原Actor等价、共同参数、接触梯度/权限、真实分段命令、joint logprob、解析GAE及新进程完整物理/空Adam恢复已核；新V已剔除错误复用的action-Q目标/slot输入，未发生训练影响。

**当前下一步P1**：固定10 train父组/12回合完整mean/sample及同态release reference；reward排序；每臂4×1024真实scratch更新、非空Adam恢复及全成本。通过后才P2/P3正式各32k，按合同连续推进，不逐段请批。P1/正式物理效果尚未PASS，不借进展报告宣称学会。正式E/I优化器更新0、Dev任务0、sealed未读，旧H0/QA2与全部历史负例不重判。GPU0只串行，不涉及Student、原FB长跑或共享资产修改。
