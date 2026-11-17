# BFM-CTFT v1：A2追加工程尝试批准记录

2026-11-17，Yi明确回复“批准”。批准对应上一条完整交付及已经发布的
[A2最小追加尝试](BFM_CTFT_V1_AOT_REPAIR_REQUEST_A2.md)，版本a1bfa3548dfee143dda93ebd729445c060c303bc。

状态：**APPROVED**。原待批稿与两次失败完整保留，不追改为成功。

唯一修订为在lazy编译前设置`torch._functorch.config.donated_buffer=False`；
`compile=True`、全部loss/GP/FP32梯度累加顺序/五Adam/物理/概率与恢复门不改。
授权一次重试及通过后同配置新进程读取、固定batch更新等价检查。
不代表接触全链S0或正式U/T已通过，不提前发车。

限额沿原A2草案：1800job秒、12GiB临时输出，计入原S0/stage/项目总账；
本次准备、导入、失败、IO、恢复及报告均计费，不另重置。若仍失败即停报，
不转eager、不改数学或试别的flag。只通过后恢复原授权S0其余资格。

存盘资格仅针对本次实际模型、五Adam、targets、normalizers、z-buffer、
RNG、实际源回放与下一更新；不冒充尚未通过的伙伴Task/solver完整训练恢复。
