# v1.1入口工程记录

首次新runtime 3f05b0c 冷启动期间，静态复核发现v1.1 observer未继承原Task的_dual_track_terminal标记，prepare可能再次包装，恢复时可能保留旧observer累计历史。未据此生成任何科学资格结果。主动TERM后未及时退出，核验owned PID后KILL；保留sidecar/日志。131.497193秒计入原预算；counter actual=0/reserved=1，未形成完整资格物理轨迹，不能保证预留中的原生tick完全未进入动力学，不虚称全部物理调用为零。

修复仅给已包住原terminal wrapper的observer加标记，防止重复包装；不改reward、动作、数据或门。新增synthetic callback binding测试10行，重装后旧observer不再计数且只读一份原callback，PASS。此为接线测试不是物理资格。冻结资格按相同seed、父组和噪声重新执行；旧失败输出保留。

## 第二次入口失败：隔离目录辅助库缺失

runtime f53e3cdaec70d71208301b65021bcc2026f97c27 的 sample-02 因新stage遗漏 pointer_layout.so/json，在加载恢复工具时退出，wall 131.879758秒，counter actual=1/reserved=1；五个采样资格回合尚未执行，没有学习更新。保留原日志及sidecar，不视为物理资格失败。

从原已验证stage复制相同辅助库及布局表，CPU导入复验PASS：MuJoCo 3.3.3、结构大小163056、173字段、CUDA未初始化。so SHA256 8e7d74e1ffb947f1ccfc82b083dbf91b6433e2da847c29dc8e96ef1709dfb439；JSON SHA256 fa3c671ca8162ed1917bf0f7fbdcf2b701aa18f41342b202dbe5a554708df5ff。新入口增加文件哈希前检，避免冷启动后再发现缺失。没有更改恢复合同、科学门或训练参数。下一次冻结资格上限1400秒，仍计入原资格3000秒分账。
