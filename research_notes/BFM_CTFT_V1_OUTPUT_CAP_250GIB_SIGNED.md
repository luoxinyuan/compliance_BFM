# BFM-CTFT v1 输出帽250GiB追加批准

Yi原话：**“如果必要的话就扩大到250GiB吧”**。状态：已批准并归档，2026-11-18。

必要性：原完整存盘方案的保守预测约232.48GiB，超过此前200GiB；另加8GiB训练轨迹分配后保守约242.48GiB，仍在250GiB内。见[Issue批准记录6967133786](https://github.com/GaoE05/robot-research-bridge/issues/67#issuecomment-6967133786)与 `bfm_ctft_output_cap_250GiB_signed_v1.json`。

仅阶段新增输出帽改为250×2^30bytes；40job小时、16M全流程physics ticks、项目240h、100GiB共享盘自由空间及已签科学/恢复/物理合同均不变。原120→200记录保留。

按固定有限N公式，最大可行共同N为131072/臂；不是根据学习效果调N，亦不是批准240h以外资源。首次正式16k须再次检查实际增长；不能靠删Adam/replay或旧失败履约。
