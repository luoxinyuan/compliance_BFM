# M39-B：原始命令保持 vs BFM动作保持

这4条都是本轮一次封存测试保存的真实MuJoCo qpos，25fps、正常1×，不是参考motion或目标关节动画。没有为视频重跑/挑seed。全程显示接触前、真实接触及5秒释放恢复；仿真时间约3.005–19.3s，更早的站立预热不在片中。实际总force cap仍5N，未做20N试验。

- **01-zero-demand.mp4**：双点保持连接、伙伴无移动/旋转需求。上排A raw，下排B action；每排左侧固定斜视、右侧俯视。看无请求漂移，不能把身体完全静止作为硬门槛。
- **02-left-and-right.mp4**：前16.32秒左+10.5°，后16.32秒右−10.5°，两个完整独立回合。上排A、下排B，各自始终同一个checkpoint；没有按方向拼模型。看有限双向转腰，不代表迈步转身。
- **03-continuous-transition.mp4**：前移25.9cm期间加入+6.2°再撤销，固定俯视四格：左上共同初始化，右上A raw，左下B action，右下T30。B误差8.90→6.73cm，但进入时仍偏晚/方向不协调，撤销后仍有残余yaw；不是统一成功展示。
- **04-straight.mp4**：右腕80.5cm牵引，四格身份同03。初始/A/B/T30接触误差分别5.70/5.85/5.73/6.91cm，实际力RMS3.51/3.59/3.45/4.27N。保留正常可接受行走，不能用它证明组合技能已解决。

图例：C青色为真实接触点；A红色为伙伴anchor；黄色箭头为实际各点力；绿色为目标heading，蓝色为骨盆heading/实际root轨迹。标题显示各点XYZ实际力与净yaw moment。active=False时显示的几何距离不是任务跟随失败；释放后允许自然收步。

## 模型与运行

这些是特权Expert，不是新Student。冻结BFM/PD，原T30/S0保留。
- common initial SHA：72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a
- A raw best64：cce7e0fedcd4bb3c6e821d933c06d29c96ebb12d4473a64cd03f5df072cd3aac
- B action best56：49cc22b8df7f08ea4b57125f5366a19e32856f9ec6d38d14f5aea7c4364427fe
- T30：c71fcfaa10d045512c0ac8afe1087b80246087798436dd73e32d1b6d73ab9f33
- 训练runtime：11b40db7897580916d55da0afaaea7e8ab646d33；终检568a8e5e664dd12d9c7b1c6ac60c75b189196d11；渲染c6d448cef4745d891570120a0c3e973ef47b7a1e。
- 原始完整轨迹：服务器compliance_BFM/bridge-v2-owned/runs/m39b-retention-20261104/C-final-06/job/；原MP4、trace SHA及manifest位于同run根videos-final/。本地仅关键MP4+此索引。
- [Issue40](https://github.com/GaoE05/robot-research-bridge/issues/40)。

## 本地验证

全部与服务器原文件SHA对应（02为两条原片无重编码顺序拼接），ffprobe时长/分辨率正常，ffmpeg全文件解码通过；已抽帧检查实际四格身份、接触/力箭头和机器人画面。

|文件|时长秒|SHA256|
|---|---:|---|
|01-zero-demand.mp4|16.32|`2867d0dd0f66462217b78bc8d80ad8a680ec35a3df96859c8372d3db6bb0f653`|
|02-left-and-right.mp4|32.64|`8d0e0ee6b9e46dc2b0e9fc0a61b567412a8adf74da37f2eb108152d4469bfb45`|
|03-continuous-transition.mp4|16.32|`ec1116365f91e1a1b1554ef3af9d3fd081286c9c4a05c57d1bc7ca23834fd11e`|
|04-straight.mp4|16.32|`f04541ea79858152c4a8fcc13e55c7f8e81c3268bbee9fbaad128e316a84c013`|
