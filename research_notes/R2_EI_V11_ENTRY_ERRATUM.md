# v1.1入口工程记录

首次新runtime 3f05b0c 冷启动期间，静态复核发现v1.1 observer未继承原Task的_dual_track_terminal标记，prepare可能再次包装，恢复时可能保留旧observer累计历史。未据此生成任何科学资格结果。主动TERM后未及时退出，核验owned PID后KILL；保留sidecar/日志。131.497193秒计入原预算；counter actual=0/reserved=1，未形成完整资格物理轨迹，不能保证预留中的原生tick完全未进入动力学，不虚称全部物理调用为零。

修复仅给已包住原terminal wrapper的observer加标记，防止重复包装；不改reward、动作、数据或门。新增synthetic callback binding测试10行，重装后旧observer不再计数且只读一份原callback，PASS。此为接线测试不是物理资格。冻结资格按相同seed、父组和噪声重新执行；旧失败输出保留。
