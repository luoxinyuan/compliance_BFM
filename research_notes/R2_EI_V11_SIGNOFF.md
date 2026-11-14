# R2 E/I v1.1 签前清单与批准登记

Yi当前会话明确授权：“按issue #63的v1.1修订合同做签前清单，然后签 v1.1并启动”。签署人/研究批准人Yi；Codex执行归档，不伪造个人签名。**v1.1已批准，分阶段执行；并非全部运行门已通过。**

三份桥接正文逐字同步，保留原REVIEW ONLY字样作为来源历史；本签署附件覆盖其待批准状态。git blob与SHA256见gate_semantics_review.json。旧v1.0及其TV门失败不改。

- [x] 22门逐项注册字段、坐标、时间窗、公式、阈值、父组、证据等级、缺格、唯一失败分支和诊断上限。
- [x] yaw excursion/TV/net/range分别计算；root用路程；none不标release。归档12mean+1sample CPU复算无新物理。sample旧TV门仍FAIL，新excursion约3.73度，通过新yaw门；warning只读分解已解释。
- [x] synthetic拒绝测试：大幅转出转回、多圈yaw、root往返、真实q、support、slip、同态root比较、U6/11度、Q四父组/70%不足、probe无正类。synthetic不当物理证据。
- [x] G11无真实同态失败例、G13整父组失败例和G14真实U角度失败缺证显式保留。
- [x] G17 Q资格及G22 BI增量明确PENDING_UNMEASURED；初始化负Q结构不等于真实Q更新/排序通过。
- [x] 保持尺度原锁.015/.030rad；全新U与std=.01/clip=.06、reward、物理、seed/父组不变。
- [x] 失败池3600s/1GiB/18000ticks包含在原16h/12GiB内；物理诊断≤2×3s、job≤600s。
- [x] GPU0实时空闲；共享盘约217GiB，单重任务串行。
- [ ] 剩余5个采样资格、真实同态恢复reference、完整增量checkpoint新进程续采样和四块Q-only资格尚未通过；按原顺序实测，绝不提前Actor训练。

**准许启动的具体工作：** v1.1存档回归及剩余5个已锁sample回合。若遇硬门失败，停止并仅执行其唯一诊断，随后报告。预算继承5544.704秒；本次准备/签署成本另计，不重置。Sealed不读不跑。
