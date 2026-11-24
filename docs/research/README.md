# 调研材料索引

本目录是**原始调研材料**（含核验链接、论文原文、issue 证据），不是操作手册。
操作请回到仓库根目录的教程。

| 文件 | 内容 | 建议阅读顺序 |
|---|---|---|
| [`FORGE_技术调研报告.md`](FORGE_技术调研报告.md) | FORGE 论文（RA-L 2025）与 Isaac Lab 实现的逐项对照、源码级参数、上游 issue 清单、复现门槛 | 1（阶段二必读） |
| [`forge/`](forge/) | FORGE 论文原文：`forge_paper.pdf` / `.html` / 纯文本、GitHub issue 检索结果 | 需要查原文时 |
| [`curobo.md`](curobo.md) | cuRobo 版本谱系 / 8 GB 可行性 / **插接对照组实验设计（E1–E6）** / 替代方案对比 | 2（阶段二 cuRobo 部分） |
| [`cuRobo_IsaacLab_技术调研.md`](cuRobo_IsaacLab_技术调研.md) | cuRobo 更细的版本、依赖、显存实测与踩坑清单（与上一份互补，有重复） | 3（安装前） |
| [`RLinf_技术调研报告.md`](RLinf_技术调研报告.md) | RLinf 架构（M2Flow）、Isaac Lab 支持实情、8 GB 判定、接入路线与工作量 | 4（阶段三路线 B） |
| [`residual_compliance.md`](residual_compliance.md) | 残差 RL 文献与柔顺控制（阻抗/导纳/混合、peg-in-hole 力学、Isaac Lab 组件） | 5（阶段三方法章） |

## 使用原则

1. **本目录里的数字都附了来源**；没有来源的标了"未确认/估算"。写论文时只引用有来源的。
2. 与本机相关的版本事实（Isaac Sim 5.1.0 / Isaac Lab 2.3.0 / torch 2.7.0+cu128 / 4060 8 GB）来自 `../../依赖安装与排错记录.md`。
3. 与本机源码相关的行号/字段，均在 `../../IsaacLab`（v2.3.0）里逐条核对过；换版本后要重核。
