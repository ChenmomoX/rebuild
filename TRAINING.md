# 训练与可视化指南

原版 `train.py` 训练完只打印指标、不保存任何东西。本目录在其基础上新增了可复现的
训练入口 `run_experiment.py`（固定种子、存模型、记日志）和可视化脚本
`plot_results.py`，原 4 个代码文件未做任何改动。

## 准备

把 `data.py` 需要的 `.npy` 数据文件放在**运行命令的工作目录**下（相对路径加载），
文件与数据集的对应关系见 [DATA_SOURCES.md](DATA_SOURCES.md)：

| 数据集 (`--dataset`) | 需要的文件 |
|---|---|
| `phy` | `physiobank_clean.npy`、`physiobank_contaminated.npy` |
| `semi` | `reference_Semi-simulated_EOG.npy`、`signal_Semi-simulated_EOG.npy` |
| `EMG` / `EOG` / `hybrid` | `EEGdenoising_EEG.npy`、`EEGdenoising_EOG.npy`、`EEGdenoising_EMG.npy` |

## 本地冒烟验证（CPU 即可，几分钟）

```bash
python run_experiment.py --dataset semi --epochs 2   # 最小数据集，验证全链路
python plot_results.py results/semi_e2_seed42_*      # 生成曲线图/时域图/频谱图
```

## 服务器完整训练

1. 上传整个项目目录（或 `git clone`），确认所需的 `.npy` 文件在工作目录；
2. 安装依赖（GPU 版 torch 按服务器 CUDA 选择 index-url）：

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install numpy scikit-learn tqdm einops matplotlib
   ```

3. 训练（建议 `nohup` / `tmux`，100 轮论文同款配置）：

   ```bash
   nohup python run_experiment.py --dataset phy --epochs 100 > train.log 2>&1 &
   tail -f train.log   # 每轮打印指标和 ETA
   ```

4. 训练结束后出图：

   ```bash
   python plot_results.py results/phy_e100_seed42_YYYYmmdd_HHMMSS
   ```

## 输出说明

每次运行生成独立目录 `results/<dataset>_e<epochs>_seed<seed>_<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 全部超参 + Python/torch/numpy 版本 + 设备（复现凭据） |
| `metrics.csv` | 每轮 train_loss / val_loss / RRMSE_t / RRMSE_s / CC / SNR / 耗时 / 是否最佳 |
| `checkpoint_best.pt` | 验证 SNR 最高的模型权重（含 epoch、指标、参数），**论文指标以它为准** |
| `checkpoint_last.pt` | 最后一轮权重（断点续训/对比用） |
| `denoised_samples.npz` | 用最佳权重对验证集前 N 个样本的去噪输出（noisy/clean/denoised） |
| `curves.png` | 2×2 训练曲线（loss / SNR / CC / RRMSE），红线标记最佳轮 |
| `time_domain.png` | 3×3 时域对比：带噪/干净/去噪三线叠加，标题含每个样本 SNR 提升量 |
| `psd.png` | 同批样本单边幅频谱对比 |
| `best_metrics.txt` | 最佳轮指标汇总 |

## 复现说明

- 种子通过 `--seed` 控制（默认 42），固定了 numpy/torch 随机数与 cudnn 确定性模式；
  数据划分本身固定（`random_state=42`）。同硬件同版本下重跑结果一致，跨硬件为同量级。
- `phy` 数据是用 `build_physiobank.py` 重建的，与论文作者的预处理可能存在差异
  （见 DATA_SOURCES.md），指标对比请以定性/同量级为准。
- 常用参数：`--mask-ratio 0.1`、`--lr 1e-3`、`--batch-size 128`（默认均与原代码一致），
  `--save-samples` 控制导出的验证样本数，`--num-workers` 在 Linux 服务器上可设 2~4 加速。
