# satellite-agent · 作物分类训练流程

用小模型（ResNet-18，约 11M 参数）在 Sentinel-2 多光谱影像上做作物 / 土地覆盖分类。
骨干网络用 TorchGeo 提供的 **SSL4EO-S12 MoCo 预训练权重（13 个 Sentinel-2 波段）**，
训练分两段：先冻结骨干只训分类头，再用较低学习率整体微调。单张 RTX 5070（12 GB）足够。

```
cropcls/
  bands.py     Sentinel-2 波段名与顺序（与预训练权重对齐）
  data.py      EuroSAT / 自有 GeoTIFF 目录数据集 + DataModule
  model.py     TorchGeo ResNet（波段子集自动裁剪第一层卷积）+ 时序 Transformer 头
  module.py    LightningModule：GPU 上做缩放/增强/resize，冻结→解冻调度
  train.py     训练入口（训练完自动在 test 集评估并写报告）
  evaluate.py  对任意 checkpoint 重新评估
  predict.py   对一批 GeoTIFF 输出预测 CSV
  download.py  预下载数据集与权重
  prepare_pastis.py  PASTIS（法国作物类型）→ 地块 patch
tests/         合成数据的快速 CPU 测试
```

## 1. 环境（RTX 5070 / Blackwell）

RTX 50 系是 Blackwell 架构（sm_120），**必须用 CUDA 12.8 及以上编译的 PyTorch（2.7+）**，
老版本会报 `no kernel image is available for execution on the device`。
显卡驱动需 ≥ 570（Windows 用最新 Game Ready / Studio 驱动即可）。

```bash
conda create -n cropcls python=3.11 -y && conda activate cropcls
# 先装 CUDA 12.8 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 再装其余依赖
pip install -e ".[dev]"

# 检查 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# 期望看到 ... True NVIDIA GeForce RTX 5070 (12, 0)
```

Windows 原生也能跑；如果 DataLoader 很慢或出现多进程报错，把 `--num-workers` 调到 2–4，
或者在 WSL2（Ubuntu）里跑，体验更接近 Linux。

## 2. 快速跑通：EuroSAT 基线

EuroSAT 是 27,000 张 64×64 的 Sentinel-2 L1C 影像块，13 个波段、10 类，
其中 AnnualCrop、PermanentCrop、Pasture、HerbaceousVegetation 与农业相关。

```bash
# 下载数据（约 2 GB）和预训练权重（约 43 MB）
python -m cropcls.download --dataset eurosat --root data/eurosat

# 训练：10 类，13 波段，默认 30 epoch（前 5 epoch 冻结骨干）
python -m cropcls.train --dataset eurosat --data-root data/eurosat

# 只训农业相关的 4 类
python -m cropcls.train --dataset eurosat --classes agri --name eurosat-agri
```

想先花 1 分钟确认环境没问题，可以用 100 张图的迷你版：

```bash
python -m cropcls.train --dataset eurosat100 --download --epochs 3 --freeze-epochs 1 --batch-size 16
```

每次训练输出到 `runs/<name>/`：

| 文件 | 内容 |
|---|---|
| `checkpoints/best-XX.ckpt` | 验证集 macro-F1 最好的模型 |
| `metrics.csv` | 每个 epoch 的 loss / acc / 平衡精度 / macro-F1、学习率 |
| `test_report.json` | test 集总体指标 + 每类 precision / recall / F1 |
| `confusion_matrix.csv/.png` | 混淆矩阵 |
| `data_summary.json`、`args.json` | 数据划分、类别样本数、全部参数（可复现） |

## 3. 用你自己的作物数据

把带标签的影像块按类别放进文件夹（每个 .tif 是一个多波段 patch，值为 Sentinel-2 DN，
即反射率×10000；如果是 0–1 反射率，加 `--scale-to-dn 10000`）：

```
data/mycrops/
  maize/  a.tif b.tif ...
  rice/   ...
  wheat/  ...
```

也可以自己分好 `train/ val/ test/` 三个子目录（每个下面再按类别分）；
否则会按类别分层随机切成 70/15/15。

```bash
# 例：Sentinel-2 L2A 导出的 12 个波段（无 B10），作物类别不均衡时加 --class-weights
python -m cropcls.train --dataset folder --data-root data/mycrops \
    --file-bands l2a --bands l2a --class-weights --name mycrops-l2a

# 只用部分波段（第一层卷积会自动只保留对应波段的预训练权重）
python -m cropcls.train --dataset folder --data-root data/mycrops \
    --file-bands l2a --bands B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12
```

`--file-bands` 是文件里**实际存储的波段顺序**，`--bands` 是喂给模型的波段。
可用预设：`all`（13 个 L1C 波段）、`l2a`（12 个，去掉 B10）、`rgb`，或逗号列表。

从 Google Earth Engine 导出时，按 `B1,B2,...,B8,B8A,B9,B11,B12` 的顺序选波段即为 `l2a` 预设。

## 4. 评估与预测

```bash
python -m cropcls.evaluate runs/<name>/checkpoints/best-XX.ckpt          # 重新评估 test 集
python -m cropcls.predict  runs/<name>/checkpoints/best-XX.ckpt path/to/tifs \
    --file-bands l2a --out predictions.csv                                 # 输出 top-3 类别与概率
```

## 5. 常用参数与 5070 调参建议

| 参数 | 默认 | 说明 |
|---|---|---|
| `--arch` | resnet18 | 也可 `resnet50`（25M，显存够，但慢约 2.5 倍） |
| `--weights` | SENTINEL2_ALL_MOCO | 也可 `SENTINEL2_ALL_DINO` 等；`none` 为随机初始化 |
| `--img-size` | 224 | GPU 上把 64×64 放大到预训练尺寸；改 128 约快 3 倍，精度略降 |
| `--batch-size` | 128 | ResNet-18 + bf16 + 224 在 12 GB 上可到 256 |
| `--epochs` / `--freeze-epochs` | 30 / 5 | 前 N 个 epoch 只训分类头 |
| `--lr` / `--backbone-lr-mult` | 1e-3 / 0.1 | 分类头学习率；骨干 = lr × 0.1，解冻后单独 warmup |
| `--class-weights` | 关 | 按训练集类别频率反比加权 loss |
| `--patience` | 0 | 验证 F1 若干 epoch 不涨就早停 |
| `--compile` | 关 | `torch.compile`，Linux/WSL2 上可再提速 |
| `--precision` | bf16-mixed | Blackwell 原生支持 bf16 |

增强只用随机翻转/转置（遥感影像没有固定朝向），全部在 GPU 上完成；
冻结阶段 BatchNorm 统计量保持预训练值不变。

## 6. 测试

```bash
pytest -q    # 合成 GeoTIFF，CPU 上几秒跑完，不需要下载
```

## 7. 欧洲作物类型：PASTIS（法国，18 种作物）

[PASTIS](https://github.com/VSainteuf/pastis-benchmark) 是法国 4 个 Sentinel-2 瓦片上的
2,433 个 128×128 时序块（2018-09 到 2019-11，每块 33–61 个日期，L2A 10 波段），
约 12.4 万个地块有作物类型标签：草地、软冬小麦、玉米、冬大麦、冬油菜、春大麦、向日葵、
葡萄、甜菜、冬黑小麦、冬硬粒小麦、果蔬花卉、马铃薯、豆科饲草、大豆、果园、混合谷物、高粱。
官方 5 折是空间上分开的，默认 fold 1–3 训练、4 验证、5 测试。

```bash
# 1) 下载并解压 PASTIS（只要 S2 部分，约 29 GB）
#    https://zenodo.org/records/5012942  ->  data/PASTIS/
# 2) 每个地块切一个 32×32（320 m）的 patch，地块外像素置 0
python -m cropcls.prepare_pastis --root data/PASTIS --out data/pastis_parcels --composite monthly
# 3) 训练（波段、时间步数自动从 data/pastis_parcels/cropcls_meta.json 读取）
python -m cropcls.train --dataset folder --data-root data/pastis_parcels --class-weights --name pastis-monthly
```

两种合成方式：

| `--composite` | 输出 | 模型 |
|---|---|---|
| `monthly`（默认） | 2019-03 到 2019-10 每月一个中值合成，8 个时间步 × 10 波段 | 时序模型：每个时间步共享同一个 ResNet-18（预训练权重），再接 2 层小 Transformer 做时间聚合，共约 12.4M 参数 |
| `median` | 整个生长季一个中值合成，10 波段 | 普通 ResNet-18 |

作物类型主要靠物候区分（冬小麦和冬大麦单看一景几乎一样），所以优先用 `monthly`。
时间范围用 `--start/--end` 调整，地块 patch 大小用 `--patch-size`，
`--no-mask-outside` 可以保留地块周边像素。时序模型默认 `--img-size 64`，
在 5070 上 `--batch-size 64` 显存很宽裕，可以往上加。

PASTIS 的类别很不均衡（草地和小麦占大多数），建议开 `--class-weights`，
并以 test 集 macro-F1 / 平衡精度为主要指标，而不是总体精度。

## 8. 下一步

- **其它欧洲地区**：EuroCrops 提供了多个欧盟国家的地块矢量和统一作物编码（HCAT），
  但不带影像。可以用 GEE 按地块导出 Sentinel-2 月合成，写成同样的
  `<split>/<crop>/*.tif` + `cropcls_meta.json`，直接复用上面的训练流程。
- **更强的时序基础模型**：Prithvi-EO-2.0-tiny（TerraTorch）原生支持多时相 HLS 输入。
