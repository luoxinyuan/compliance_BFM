# R2 v1.1 sample入口完成

runtime 47098d319ad8bf8cc860d38fc9d7c0483292e577；job v11-remaining-sample-05，561.581345秒，20501个实际5ms tick。五个回合/四个父组（stop/continue同父组）全部通过入口保护；3605次E/I控制对齐包含冻结参考分支，不全部是随机采样。无优化更新，sealed未执行。

| family | yaw excursion deg | slip peak m/s | recovery path m | reference path m | gate |
|---|---:|---:|---:|---:|---|
|near|2.301397|0.008095|0.036647|0.012358|PASS|
|mid|3.116592|0.016363|0.059324|0.033120|PASS|
|zero|3.425195|0.007892|0.029157|0.001171|PASS|
|stop|2.288565|0.011699|0.055320|0.033619|PASS|
|continue|2.635134|0.016156|0.059996|0.033482|PASS|

全部真实关节越界和无支撑tick为0，5次完整分支有效状态/RNG逐值恢复，stop/continue前缀哈希相同。TV高仅按新合同警告解释，不等于大幅自转。none沿用已独立重算存档；无contact的none不计恢复。保持尺度仍由原10父组mean标定，未用本次4父组重定。

## 接续范围
四块×250真实50Hz的Q-only入口，E/I串行，Actor/U更新0；每块Q4组，I从第2块起表示2组。继承原全新初始化、锁定seed/优化配置/父组；当前只资格化checkpoint、吞吐与恢复，不能称Q排名已通过。至少3500条和完整覆盖/跨父组排序门仍待后续。增量存盘沿无损core-delta；新进程读回需逐值物理/RNG/Adam检验。未通过不得扩大训练。
