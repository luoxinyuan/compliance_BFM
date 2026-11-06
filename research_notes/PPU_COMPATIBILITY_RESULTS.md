# PPU兼容性检查 — 2026-11-06

结论：真武PPU-ZW810E已实际通过当前原BFM的前向、latent梯度、I/C0同状态输出及短程真实MuJoCo闭环检查。它可以作为后续实验的候选运行后端；尚未执行Issue47接管/PPO，也未验证PPO并行采样、训练吞吐或跨平台长回合等价性。

## 身份与隔离

- 原运行源码：`cc5765522826cbe130635321f0a3056362faa184`，以只读Git archive复制。原服务器的实验worktree、BFM/PD和权重均未改动。
- 完整BFM：3,384,968,272 bytes，SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`；复制后完整校验通过。没有下载新模型/replay buffer。
- I：`72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`；C0：`ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c`。两者均逐文件校验。
- 新服务器独立根目录 `/root/compliance-ppu-20261106/`：venv、原源码快照、测试脚本、sysroot和output。约963MiB。
- 已有模型、motion等私有资产复制至 `/mnt/workspace/compliance-ppu-20261106/`；该路径是CPFS网络盘，不是系统盘。旧绝对项目路径在新服务器上仅作为该独立目录的兼容链接；没有覆盖现有项目。
- 设备97920MiB，10 vCPU、约100GiB RAM。启动与结束均无其他PPU计算进程；结束0%利用率、2MiB占用。所有测试已退出，没有排队或后台训练。

## 环境与可复现入口

厂商PyTorch `2.9.0`（报告CUDA兼容版本12.9）；driver `1.3.2-d7f5a2`、PPU-SMI1.22。保留厂商PyTorch，没有安装NVIDIA torch wheel。Python3.12.3、NumPy1.26.4、MuJoCo3.3.3、joblib1.5.1、hydra-core1.3.2、gymnasium1.1.1、tensordict0.8.3、safetensors0.7.0、Open3D-CPU0.19.0。与旧NVIDIA PyTorch2.7.1运行栈不同，不假设逐位一致。

只在会话中加载厂商SDK；缺少的GL分发库用Ubuntu包解到私有sysroot，没有apt安装或修改系统库。新venv继承厂商site-packages，再用自己的包覆盖必要依赖，不改系统Python、shell配置或旧服务器环境。

```bash
source /usr/local/PPU_SDK/envsetup.sh >/dev/null
export LD_LIBRARY_PATH=/root/compliance-ppu-20261106/sysroot/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
/root/compliance-ppu-20261106/venv/bin/python /root/compliance-ppu-20261106/ppu_tensor_check.py
# 新attempt目录必须不存在，历史输出不覆盖：
PPU_CHECK_ATTEMPT=physical-03 /root/compliance-ppu-20261106/venv/bin/python /root/compliance-ppu-20261106/ppu_check.py
```

入口脚本副本：`ppu_compatibility_tensor.py`、`ppu_compatibility_physics.py`。测试仍使用原BFM/PD及原200Hz物理、50Hz actor、10Hz高层；BFM实际推理及梯度均在PPU的`cuda:0`兼容设备。沿用原CPU装载完整权重、仅将推理模块移到加速器的方式，不是CPU推理替代。MuJoCo物理仍按原链在CPU执行。

## 实测结果

| 检查 | 实际结果 |
|---|---|
| 18个保存的I/C0真实状态：相同BFM观测和已执行z | action最大绝对差 `5.66244e-7` |
| 同状态原映射q_PD | 最大绝对差 `2.73208e-5°` |
| 18个保存的高层输入：冻结I/C0均值raw | 最大绝对差均 `2.23517e-8` |
| 通过原冻结actor回到临时latent的梯度 | 有限且非零，norm `0.00650458`，设备`cuda:0` |
| 原BFM参数/推理权重 | 物理检查前后hash及参数version未变，没有BFM梯度或优化更新 |

新建兼容性回合：原standing latent；3s无力稳定后一次性对齐真实右腕；+X3.5cm、2s smoothstep、6s释放、8s结束；K80/D8、5N cap。它不是旧C0/I长距性能复测，也不是新科学任务成功。

| 物理指标 | 结果 |
|---|---:|
| 完整记录物理步 / 仿真时长 | 1600 / 8.0s |
| 实际接触记录步 | 600 |
| 接触期3D误差RMS | 1.968cm |
| 实际力RMS / peak | 1.618 / 2.049N |
| root最低高度 / pelvis最大tilt | 0.7797m / 2.7901° |
| 关节越限 / 力矩饱和 / 安全终止 | 0 / 0 / 无 |
| 8s物理段wall | 9.583s，非RL采样吞吐基准 |
| PyTorch峰值allocated | 174.54MiB，不等于驱动全部内存占用 |

原始数值结果见 `ppu_compatibility_artifacts/`；完整私有物理qpos/日志见新服务器`output/physical-02/`。Mac摘要与脚本：`~/Projects/compliance-bfm-diagnostics/ppu-compatibility/`。没有为兼容检查重复制作行为视频。

## 保留的失败与修复

1. 未加载SDK时找不到`libhggcrt1.so`：库存在，按厂商会话环境加载后解决；没有修改全局环境。
2. PyPI文件下载超时：改国内镜像，非硬件失败。初次非固定matplotlib依赖试图升级NumPy，随后固定为1.26.4/contourpy1.3.2/matplotlib3.10.0，完整运行使用固定版本。
3. 第一版梯度探针调用原`model.act()`：其内部`actor()`本来有`no_grad`，故没有梯度。改用已有LPP方式`model._actor(model._normalize(obs), z, std).mean`后通过；这是测试代码入口错误，不是原PPO或PPU梯度故障。
4. 缺少环境依赖及`libGL.so.1`：在独立venv/sysroot补齐，未修改BFM代码。
5. `physical-01`在读取原motion时`KeyError:72`：预装joblib1.1.1无法正确读取此数据；独立venv对齐原环境1.5.1后，`physical-02`完整通过。初次失败及日志保留。

## 范围与下一步

这是用户新授权的硬件迁移验证，0新增研究RL决策、0新增Student/Expert监督更新、0正式PPO任务。Issue47原预登记、封存条件及模型保持不变。首次兼容检查不是新的C0恢复证据，也没有证明PPU与NVIDIA长程接触动力学逐位相同。

若后续继续Issue47，先在这一后端运行冻结C0/I原任务回归，核查完整历史/reset与同后端分支前缀一致性，再执行原授权接管；不能直接把旧NVIDIA的逐位前缀断言静默放宽成科学等价，也不能把新后端第一次运行的差异归因于策略学习。PPO多进程/优化器状态与采样吞吐尚未资格化。本次完成后停止，没有自动发车。
