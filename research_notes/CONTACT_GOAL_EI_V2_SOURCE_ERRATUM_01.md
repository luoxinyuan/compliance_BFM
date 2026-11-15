> 后续状态：Yi已追加授权自行修复源实现后继续；本文保留修复前失败事实，非最终停止。见[SOURCE_FIX_AUTHORIZATION_01](CONTACT_GOAL_EI_V2_SOURCE_FIX_AUTHORIZATION_01.md)。

# Contact-Goal E/I v2 — Source-Ancestry Erratum 01

2026-11-15；状态：CONFIRMED，影响限于v2新source展开/资格，不重判H0及旧训练结果。

**原主张/设计：** v2继承34/16/28父组和原source_parent_id，把train/dev各6组切为arm-ready，仍要求源祖先互斥。

**问题：** 原ID编码了standing实际使用的pulse/phase参数；切为goal_arm5后这些字段未参与执行，arm_fraction又均为1。不同ID不能继续证明不同真实源。两个train完整真实前缀逐值相同，五对train/dev源有效配方及结束时刻重合。

**隔离：** expanded-train/dev锁保留为失败设计证据，状态BLOCKED_G01_DATA_INSUFFICIENT，禁止供正式训练/独立dev声明使用。实际两个TRAIN原生审计归档只用于源资格证据，不是训练数据。没有E/I正式checkpoint/Dev/Sealed产物需追溯隔离。

**处理：** 不改获批合同原文，不换manifest、不隐藏失败；P0停止，待明确源生成/祖先修订。不是通过删除源、改独立分母或放宽判据的工程修复。

**附工程失败：** 首次审计导入缺stage相对helper，按§9复制同SHA已验证ABI helper后新输出重试成功；它未造成或修复本源分组问题。

完整证据、边界、成本：[P0报告](CONTACT_GOAL_EI_V2_P0_SOURCE_RESULTS.md)与[机器摘要](contact_goal_ei_v2_p0_results.json)。此erratum不声称旧standing源/旧H0均存在同样问题。
