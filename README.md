# ccf-c-detector-efficiency — 通用检测器边缘效率基准 + 选型指标

> **Public reproducibility archive (v0.2.1).** DOI: [10.5281/zenodo.22839651](https://doi.org/10.5281/zenodo.22839651). This version contains the verified COCO annotation provenance, complete-val selective INT8 confirmation, reproducibility scripts, model hashes, and curated release metadata.

> **第二篇论文项目**（SCI 期刊线，CCF-C 类；主投 **Machine Vision and Applications**，内部目标投出 ~2026-11 中）。
> 与已投 SIViP 的 SCI 项目 `d:\Article\neu-det-project` **完全隔离**——本目录独立，**绝不动** SCI 的任何文件（SCI 可能还要修补）。

---

## 一、这篇论文做什么（一句话）

在**同一个 CPU 台架**上系统测量 YOLO(CNN) 和 RT-DETR(Transformer) 两族检测器的真实延迟/参数量/FLOPs/体积/INT8 量化表现，并把 LAE 保留为一个**描述性**的联合预算查表指标；核心证据是可复现的 CPU 测量协议、FLOPs 与真实延迟的偏差，以及 INT8 导出图与实际执行内核的关系。

## 二、为什么能中（核心卖点）

1. **空白**：现有效率基准几乎全在 GPU(T4) 上测、或只测 YOLO 一族；"YOLO + RT-DETR 同一 CPU 台架的效率分级"没人做过。
2. **不是纯测速报告**：量化 FLOPs 与真实 CPU 延迟的偏差，并检查 INT8 导出图、算子融合和运行时行为；LAE 只作透明的描述性分析，不包装成已验证的新方法。
3. **测量公平性**：RT-DETR 免 NMS、YOLO 带 NMS，对比有偏差——我们修正它。

## 三、目录结构（每个文件夹干什么）

```
ccf-c-detector-efficiency/
├── data/            # 权重、COCO 数据、探针产出（大文件不入文）
├── scripts/         # 效率测量 + 出图脚本（Python）
├── paper/
│   └── manuscript/  # 手稿：按节拆开的 .md（0_abstract 到 6_conclusion）
├── results/
│   └── data_ledger.md  # ★数字账本：所有数字的唯一来源（绝不编造）
├── release_data/       # 公开发布的精简 CSV、元数据、模型哈希和环境摘要
├── notes/           # 调研/头脑风暴记录
└── README.md
```

## 四、实验设计（极简版）

| 项 | 决定 |
|---|---|
| 数据集 | COCO val（只测前向延迟/精度，**不训练**） |
| 模型 | YOLO11n、YOLOv8 n/s/m/l（CNN）+ **RT-DETR l/x（Transformer，Ultralytics 官方，同库同源）** |
| 测量 | CPU 实测延迟 / 参数量 / FLOPs / 体积 / INT8 量化 |
| 指标 | 实测端到端延迟、FLOPs/延迟比、INT8 执行图；LAE 仅作描述性敏感性分析 |
| 产出 | 4 张图 + 选型协议（决策树/查表） |

## 五、台架纪律（已拍板 2026-09-04）

**CPU 延迟台架 = 本地 Windows（Intel Core i7-14650HX，16 核 24 线程，16GB，torch 2.13.0+cpu）**
**框架 = 全部 ONNX + onnxruntime CPU**（YOLOv8 + RT-DETR 统一导 ONNX、统一 ORT 测延迟 → 两族计时口径公平，顺带统一 NMS 口径，也契合"边缘部署栈"主题）

⚠️ 红线：
1. **CPU 延迟必须在纯 CPU 台架上测**——A10 是 GPU，**绝不能拿它测 CPU 延迟**（审稿人一眼看出造假）。
2. **全篇同一台 CPU**（本机 i7-14650HX），注明 CPU 型号/线程数；跨台架不混数值。
3. FLOPs ×2 对齐 ultralytics 惯例；延迟报稳定口径（预热 N 次取平均）。
4. 每个数字进 `results/data_ledger.md`，标来源/口径/台架。

## 六、进度

- [x] W1 骨架：目录结构 + README + 数字账本模板 + CPU 台架定案（i7-14650HX）
- [x] W1 跑通 YOLOv8/RT-DETR 依赖（torch/ultralytics/onnxruntime），双模型 ONNX 导出
- [x] W2 测量协议：16 线程 + 端到端 NMS 口径（真实 COCO 输入）
- [x] W2 探针实验（YOLOv8n/l + RT-DETR-l）→ 效率层级错位成立（19.0>13.7>8.6）
- [x] W3a RT-DETR GFLOPs 校准 92.54→105.60（运行时计数，transB 修复，skip=0）
- [x] W3b 导出全 6 模型 ONNX（yolov8n/s/m/l + rtdetr-l/x）
- [x] W3c 全 6 模型正式 measure → 账本 §1/§2（发现窗口漂移）
- [x] W4a 决策：延迟硬化协议=**五轮交错中位数之中位数**（旋转模型顺序，已生成 raw/summary/metadata）；完整 COCO val2017 已用标准 `pycocotools` COCOeval 统一重评估，结果进入 `results/full_coco_map_official.csv` 和 `release_data/full_coco_map.csv`
- [x] W4b INT8 量化收官（账本 §3 + notes/05）：三配方（dynamic/naive static/selective）+ 两发现
  （①输出 Concat 混装→mAP≈0；②RT-DETR static 崩溃=跨族观察）；体积 / 同窗口延迟Δ /
  500 子集 mAPΔ 齐备（static ×2、dynamic 7.8-9.4×、selective 精度 ±0.009 无损）
- [x] W4b.5 收尾：README 同步（本表）+ 频率数据入账（修正① 措辞待写作时用户定，见账本 §3 频率行）
- [x] 修正① 开发阶段稳健性记录：YOLOv8n/m + RT-DETR-l 隔日 3 轮 **Rank(Day1 vs Day2) 完全一致**（+5%/−5%/−13% 漂移但秩不变）→ 已入账本 §2；该记录不替代新版七模型交错重测
- [x] 修正② mAP 双轨分工：统一全量 val2017 当 LAE/Pareto 基准 / 500 子集只算量化 Δ；全量结果已入公开 `release_data/`
- [x] 投稿稳健性增强：官方 COCO 标注归档与当前 JSON 已按大小/MD5/SHA-256 核验；四个 detection-head-preserving YOLO INT8 工件已完成 5,000 图官方 COCOeval 点估计并入 `release_data/full_coco_int8_selective.csv`
- [x] Y 挡数据收齐：YOLO11n 导出入账本；7 模型规范同窗口表（QC 0 间隙）入账本 §2
- [x] 修正③ α×β 秩稳定 + Pareto + 朴素对照已入账本 §5：计划区 117 点 100% 恒序；被支配=v8n/v8l；
  朴素指标全同序（rho=1.0）→ "显著优于"说法被数据否证；**§5.1 措辞复核**：退化角 4 翻转点非"RT-x 跳
  第一"（(0.05,0.05) 单点 top-1=RT-DETR-l），账本/notes/06 已改准；P4 图改题已拍板（notes/06）
- [x] W5 图 P1-P3 已生成（账本 §6 + notes/07 图注；300dpi PNG+PDF 在 results/figures/）；v2 按用户目检反馈重画（P1 拆三面板/P2 图例高亮/P3 逐点标真值）；**v3 按第二轮审图意见处理**（P1 "Ratio (A / B)" 口径+延迟 whisker、P2 log 轴+点旁参数量(用户拍板)、P3 y 限余量/ORT v1.28.0 工具链措辞/FP32 绝对基线脚注）；**v4 按第三轮用户反馈处理（仅 P1）**：族配色+高亮第 4 根跨族柱（用户拍板）、删 ratio=1 旁注改 (a) 图例、图注精简、(c) 标题带台架、柱顶数值加粗上移、x 轴单行全名（测试 9/9 已清理）；v4.1 修图注水平截断（手动断 8 短行+画布加高，测试 4/4 已清理；notes/07 记 v4/v4.1 改动）；**v4.2 修 P2/P3 图注同款截断+贴边**（P2 画布 7.0×5.1/bottom 0.28/5 短行、P3 7.8×4.3/bottom 0.32/6 短行，测试 13/13 已清理；notes/07 记 v4.2）；**P4 v5 双面板新增**（秩稳健热图 + 余量失真，用户拍板改题；热图网格代码内现算=账本 §5.1 同源、图注 f-string 现算，测试 12/12 已清理；notes/07 记 v5）；**v5.1 按用户反馈改 P4 (b)**（parity=1 虚线由细灰改为深色 lw1.3 置顶并纳入图例、图例改语义标签 Theoretical/Measured CPU 并移右上空白角、图注点破 magenta=理论效率/blue=实测 CPU 效率色码，测试 6/6 已清理；notes/07 记 v5.1）；P1-P4 全 4 图 300dpi PNG+PDF 在 results/figures/
- [x] **Y 挡达成（2026-09-07）**：Kaggle 注册受 Google reCAPTCHA 阻拦 → 改用 **ModelScope 魔搭免费纯 CPU**（8 核、无 GPU；notebook 双平台通用 + 修 IMG NameError，测试 7/7 已清理）；该云端结果仅保留为开发记录；修订稿不将其作为跨平台证据（`results/kaggle_fwd.csv` 存档）
- [x] W5-6 选型协议已落稿（查表+4 步规则；LAE 降为描述性分析）；图 P1-P4 已生成，最终投稿版仍需同步五轮数据
- [ ] W6-7 写作 + 5 席自评
- [ ] W8 投 MVA 期刊（内部目标 ~2026-11 中）

## 七、MVA 完善方案 v5 执行入口

当前主张限定为本机 Intel i7-14650HX。旧的 ModelScope/Kaggle 跨平台结果保留在账本中作为开发记录，不再作为正文中的跨平台证据。

### 本机七模型交错重测

```text
python scripts/measure_interleaved_7.py --rounds 5 --warmup 20 --reps 30 --n-imgs 8 --threads 16
python scripts/thread_sensitivity_anchors.py --rounds 3 --warmup 20 --reps 30 --n-imgs 8
```

输出分别为 `results/main7_interleaved_raw.csv`、`main7_interleaved_summary.csv`、`main7_metadata.json` 和 `results/thread_sensitivity.csv`。

### 完整 COCO 准备

```text
python scripts/prepare_full_coco_val.py
```

脚本会校验压缩包 SHA-256、确认 5,000 张图片并生成 `val2017_manifest.json`；图片和权重不进入版本库。

主平台的 8 张计时图已经记录在 `results/main_timing_images.txt`。主平台默认测量仍走
`eval_common.list_images()`，不会切换到 manifest；manifest 只供第二个平台读取。
YOLOv8s 的 S2 三工件及 SHA-256 见 `results/int8_artifact_manifest.csv`，显式工件模式会在运行前
逐项验哈希，禁止自动重新导出。

在 Pi 5 仓库根目录执行以下探针命令（S0/S1 未通过时不要进入下一步）：

```text
python scripts/probe_qdq_operator.py --model yolov8s \
  --artifact-manifest results/int8_artifact_manifest.csv \
  --imgs data/coco128/coco128/images/train2017 \
  --manifest results/main_timing_images.txt --threads 4 \
  --rounds 3 --reps 30 --census-only \
  --optimized-dir results/pi5/s2_optimized \
  --out results/pi5/s2_census.csv

python scripts/measure_interleaved.py --model yolov8s \
  --artifact-manifest results/int8_artifact_manifest.csv \
  --imgs data/coco128/coco128/images/train2017 \
  --manifest results/main_timing_images.txt --platform Pi5 \
  --threads 4 --rounds 3 --reps 30 \
  --out results/pi5/s2_interleaved.csv

python scripts/analyze_cross_isa_probe.py \
  --census results/pi5/s2_census.csv \
  --summary results/pi5/s2_interleaved_summary.csv \
  --required-rounds 3 --out results/pi5/s2_decision.json
```

探针阶段（S0+S1+S2）最多 1 个工作日；只有探针通过后，才运行独立的完整 benchmark：

```text
python scripts/measure_pi_benchmark.py
```

完整阶段最多 2 个工作日，默认测 YOLO11n、YOLOv8n、YOLOv8s、YOLOv8l、RT-DETR-l 和 RT-DETR-x，
不测 YOLOv8m；输出 `results/pi5/pi5_samples.csv`、`pi5_summary.csv` 和 `pi5_metadata.json`。
脚本默认拒绝覆盖已有输出，需明确指定新的 `--out-dir` 或 `--force`。

最终导出时 `scripts/export_docx.py` 会同时检查本地 cover letter 的标题和三项贡献。
标题/摘要/贡献发生变化后，先同步 `paper/submission/cover_letter.md`，再把 Editorial Manager
文本框粘贴后的返回文本保存为临时纯文本，并执行：

```text
python scripts/diff_cover_letter.py --pasted <em-returned-cover-letter.txt>
```

只有该 diff 为空，cover letter 才算完成。
