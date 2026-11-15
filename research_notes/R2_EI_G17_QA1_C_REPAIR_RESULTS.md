# QA1 C装载修复结果

Yi已在当前会话明确授权C修复并在原子额度内跑完。原erratum的历史失败不重判；本次装载问题已通过实际回归关闭。

唯一修复：`pristine_load`将完整source深拷贝后交给原`ApprovedLearner.load`。不修改原q_update、loss、lr、EMA、sampling schedule或科学门。

CPU Adam fixture：PASS，源缓存不变、相同数学更新、后续独立装载逐值一致，最终CUDA_initialized=false。C第一实际更新组57（属于456组内）后，原source完整digest仍一致、另一个learner载入完整Q56逐值一致，RNG不变。C128/256/512非Q部分与source缓存hash均保持，C56/128/256/512完整agent包读回一致。原U包SHA只读复核，未重跑。

实际运行e1fe70b20fc48994638fade189e415e9da61dd0e；456新增C Q组、Actor/U/I更新0。最终科学结果见[完整QA1报告](R2_EI_G17_QA1_COMPLETE_RESULTS.md)，不能以工程修复PASS替代Q资格。

原失败/日志/93b5f48部分报告保留。首次CPU fixture的默认fork_rng CUDA初始化警告已改为devices=[]，最终回归不初始化CUDA；一次clean-worktree启动拒绝没有子训练/物理。成本均在480秒修复准备上界中，不漏计也不虚构实际训练。
