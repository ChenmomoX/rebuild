# 数据来源说明

`data.py` 需要的 7 个数据文件及获取方式（均已放入本目录）：

| 文件 | 对应数据集类型 | 来源 |
|---|---|---|
| `EEGdenoising_EEG.npy` (4514, 512) | `EMG`/`EOG`/`hybrid` | EEGdenoiseNet 官方仓库 `EEG_all_epochs.npy` 重命名，[ncclabsustech/EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) |
| `EEGdenoising_EOG.npy` (3400, 512) | 同上 | 同上（`EOG_all_epochs.npy`） |
| `EEGdenoising_EMG.npy` (5598, 512) | 同上 | 同上（`EMG_all_epochs.npy`） |
| `reference_Semi-simulated_EOG.npy` (54, 19, 5401) | `semi` | SS2016 数据集：Klados & Bamidis, *"A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques"*, Data in Brief 2016，[Mendeley Data v4 (wb6yvr725d)](https://data.mendeley.com/datasets/wb6yvr725d/4) 的 `Pure_Data.mat`，按最短记录长度 5401 截断后堆叠 |
| `signal_Semi-simulated_EOG.npy` (54, 19, 5401) | `semi` | 同上，来自 `Contaminated_Data.mat` |
| `physiobank_clean.npy` (12420, 1, 256) | `phy` | **重建生成**，见下 |
| `physiobank_contaminated.npy` (12420, 1, 256) | `phy` | **重建生成**，见下 |

## physiobank 数据（`build_physiobank.py`）

论文（Chen et al., *Knowledge-Based Systems* 2025, DOI 10.1016/j.knosys.2025.114703）的
"PhysioBank" 数据集对应 PhysioNet 的
[**Motion Artifact Contaminated fNIRS and EEG Data** v1.0.0](https://physionet.org/content/motion-artifact/1.0.0/)
（Sweeney et al., IEEE TBiT 2012, DOI 10.13026/C2988P）：23 段双通道前额 EEG（2048 Hz），
一个换能器保持不动作为 ground truth，另一个被人为移动制造运动伪迹。

作者未公开他们的预处理代码和 npy 文件，`build_physiobank.py` 按以下流程重建：

1. 按 EEG 触发信号截取实验区间；
2. 两通道同时以 `resample_poly` 8 倍降采样 2048 Hz → 256 Hz；
3. 每条记录用"每秒 std 的 p90/p50 比值"（间歇性伪迹指标）判定被干扰通道
   （23 条记录全部判定为通道 2，与原实验设计一致）；
4. 按 1 s 无重叠切段，**并去掉每段各自的直流分量**，得到 (12420, 1, 256) 的
   clean/contaminated 数组。

第 4 步的去直流是必须的：两个换能器电极直流电位不同，而 `data.py` 的 phy 分支
用带噪段的均值/标准差归一化干净段，不去直流时干净目标会带上数百个标准差的偏移
（实测逐段偏移中位数 354，最大 5728），导致 SNR/RRMSE 指标失真（模型以带噪信号
为目标，均值≈0，无法拟合该偏移）。去直流后逐段偏移为 0。

注意：与论文作者的预处理可能存在差异（如是否只保留含伪迹区间、滤波参数等）。

原始 CSV 压缩包 `physionet_raw/EEG-csv.zip`（约 145 MB）因超过 GitHub 单文件 100 MB
限制未纳入版本库，请从 PhysioNet 下载：
[motion-artifact/1.0.0](https://physionet.org/content/motion-artifact/1.0.0/)，
解压到 `physionet_raw/` 后修改 `build_physiobank.py` 重跑即可。
