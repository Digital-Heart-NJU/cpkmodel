# Munneke 心肺模型

本目录保存用于 2026 年 11 月 13 日心腔压力对比的负荷匹配版本。它包含四腔五壁、20 个心肌 Patch、TriSeg 室间隔力学、心包、5 个 ArtVen 血管床、8 条 Tube 和自主呼吸。呼吸力学通过胸膜压单向作用于循环。模型包含 76 个物理状态和 1 个口端气量账本状态，并另存 Tube 波传播历史。

## 运行

在 D 盘的仓库副本中，从仓库根目录执行以下 PowerShell 命令。脚本使用现有 `D:\anaconda\envs\biomodel\python.exe`，在本目录的 `runs/` 保存输出和缓存。

```powershell
& .\munneke_cpmodel\run_windows.ps1 -Check
& .\munneke_cpmodel\run_windows.ps1 -Beats 2 -Out 'runs\example'
& .\munneke_cpmodel\run_windows.ps1 -Beats 100 -Out 'runs\pressure_flow_85s'
```

默认 500 微秒使用 `data/calibrated_reference.npz`。传入 `-DtUs 250` 使用对应的半步长标定参考。每搏输出心腔压、血管压、流量、容积与检查点。`cardiac_pressure` 为心腔源压，`node_pressure` 为计算连接流量的节点压。

## 参数和示例

- `data/` 保存心肌、血管、呼吸参数及冻结参考状态。初始化以 0.85 s 心周期、约 91.5 mmHg 参考压力和 5.1 L/min 心输出量为目标。正式运行冻结负荷参数。
- `source_identity.json` 记录与历史实验完全匹配的 35 份模型源码及参数文件摘要。`manifest.json` 用于校验此目录的交付文件。
- `example/pressure_flow.png` 展示历史 85 秒运行的末一个完整射血到射血周期。上排是压力，下排是流量，竖虚线标记同一时刻。`example/` 还保存绘图对应的压力数据和运行摘要。

该版本用于研究计算，负荷匹配仅覆盖原设定的目标。肺动脉和两房压力仍有偏差，临床外部验证尚未完成。模型范围为心脏、体肺循环和自主呼吸力学。

## 来源

心肌与 ArtVen/Tube 方程参考 [CircAdapt/Munneke2023 的固定提交](https://github.com/CircAdapt/Munneke2023/tree/85d9d8bdaddb6b09902ba3578cba1721caa6e145)，在本项目用 Python 实现。呼吸力学参考 Laudenzi、Dalmaso 和 Muller 的 Cardio-respiratory-model，呼吸模块的原始许可随附于 `third_party/respiratory_LICENSE`。冻结参考和示例属于本项目计算产物。
