# Conditional adapter: actual physics videos

真实保存qpos、1×/25fps；非reference或qPD动画。固定seed4501展示，不按结果挑seed。

01：F/R各自同一checkpoint的小9cm与长65cm。四格左小右长、上F下R。短回合结束后明确标记末帧保持，不是继续物理。
02：同一个R4501，共同前缀7.05s后伙伴停止/继续。上停止，下继续。
03：同一个R4501，同22cm请求，不同真实自然源前缀。上中性，下预伸臂；不是qpos重置。

完整接触、保持和自由恢复都保留；释放后几何误差不计跟随。已保留失败。

模型身份：
- F4501: 52956080217e6346b1a8b196ad411cbc84a07e248a8e747eeedf2dcc775e12f6; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105/F4501/F4501-2000.pt
- R4501: 298af356d386f79377ca103b4fc82baecbf6e18a997a958633dade2cc7733a37; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105/R4501/R4501-2000.pt
- F4502: 9e009414dcac884f8bfc39566e5c8ea6a9ca3831f47f5e56e0a29ea0421f3462; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105/F4502/F4502-2000.pt
- R4502: d940c66409c1cec7f6e1ffec7c25589c99abf227a597c3bdb67447d4ec9d0561; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105/R4502/R4502-2000.pt
- I: 72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m39a-demand-20261104/B-ppo-03/job/best.pt
- C0: ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/prior-recovery-20261105/correct-C0/C0.pt
- C1: 8ead3da1a2844fa8ac4cae3543df7aae93021202a8d8335af2f164fd2076455d; /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/prior-recovery-20261105/correct-C1/C1.pt
- 01-same-model-small-long.mp4: 13.84s; SHA256 17d32795c883b3440a3eb0665b4457674664e7cb122448b47bdcf5cc59e195e3
- 02-common-prefix-stop-continue.mp4: 14.28s; SHA256 c40c117d3e14b0340047f6821f1b94d003ea7fad26b620fb7583896dbfa1a881
- 03-natural-source-states.mp4: 11.72s; SHA256 fd69bc6d750f01072fda07d178f9be21d715eaa66fc9b374b48b856081f9e8b2

完整指标见Issue45 / ADAPTER_RESULTS.md；理想力仿真不是实机验证。

补充指标（全接触3D误差cm/真实力RMS N）：视频01 F4501短9/长65=4.671/3.873与6.012/4.894；R4501=4.566/3.778与6.657/5.446。视频02 R4501停止/继续=5.283/4.312与6.035/4.931，7.05s前前缀完全一致。视频03 R4501中性/arm-ready22=6.494/5.334与7.486/6.129；接触onset分别4.85s/6.85s，后者包含4–6s真实自然姿态前缀，因此不能按同一屏幕时刻比较接触进度。模型未按视频条件更换；并非每项都改善。
