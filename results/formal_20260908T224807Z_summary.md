# Qwen3.5-9B 正式实验结果

## 结论

Qwen3.5-9B 在隐喻条件下生成的短续写，整体更接近恰当释义条件，而不是不恰当字面义条件。这个 RQ1 方向效应在统计上明确。

RQ2 的总体模型也得到同向结果：模型在显式问答中越偏向恰当释义，其自由续写越呈现恰当释义方向的对齐。在回归中固定只看目标词时已有的候选偏好后，语境增量的条件斜率仍为正。因此，本次结果不是显式判断与生成行为的 dissociation（分离）；更准确的概括是 **aggregate agreement with limited item-level coupling**，即总体一致，但项目层面的联系有限。

## 实验身份

- 模型：`Qwen/Qwen3.5-9B`，官方 post-trained model，RQ2 关闭 thinking 路径
- 固定 revision：`c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- 实际参数量：8,953,803,264
- 运行：单张 NVIDIA GeForce RTX 5090，BF16，无量化，eager attention
- 数据层级：880 triples、595 target items、553 source-sentence clusters
- 正式 run：`20260908T224807Z`
- 模型 run ID：`munch-qwen3p5-9b-20260908T224851Z-9fd86dafddf5`

## RQ1：下游预测对齐

项目层面的效应定义为

\[
\Delta_i=d_i(M,I)-d_i(M,A).
\]

这里的 \(d\) 是 continuation-set distance（续写集合距离）。\(\Delta_i>0\) 表示隐喻条件 \(M\) 的续写更接近恰当释义 \(A\)，而不是不恰当释义 \(I\)。

| 结果 | 数值 |
|---|---:|
| 平均 \(d(M,A)\) | 0.774173 |
| 平均 \(d(M,I)\) | 0.788438 |
| 平均 \(\Delta_i\) | 0.014265 |
| CR1 cluster-robust standard error（聚类稳健标准误） | 0.001271 |
| 95% confidence interval（置信区间） | [0.011769, 0.016761] |
| 单侧 t test（单侧 t 检验） | \(t(552)=11.227\) |
| 单侧 p value | \(8.66\times10^{-27}\) |
| 正向 / 负向项目 | 413 / 182 |

95% 置信区间完全高于 0，且 p value 远小于 0.05，所以该数值在统计上显著。平均效应约为项目间 \(\Delta_i\) 标准差的 0.45 倍；这说明方向稳定，但不应把“显著”误读为距离差非常大。

## 局部 surprisal diagnostic

辅助量定义为

\[
F_i=\operatorname{surprisal}(q3\mid I)-\operatorname{surprisal}(q3\mid A).
\]

| 结果 | 数值 |
|---|---:|
| 平均 \(F_i\) | 2.467271 nats |
| CR1 cluster-robust standard error | 0.134823 |
| 95% confidence interval | [2.202443, 2.732099] |
| 单侧 t test | \(t(552)=18.300\) |
| 单侧 p value | \(4.00\times10^{-59}\) |
| 正向 / 负向项目 | 462 / 133 |

同一个三词片段在 \(A\) 后明显比在 \(I\) 后更容易预测。这证明 \(A/I\) 条件确实包含明显的局部合理性差异；该分析用于描述材料差异，不替代 RQ1。

## RQ2：显式判断与生成行为的一致性

每个显式分数都来自两种候选顺序下的平均回答差：

\[
\log p(\text{选择恰当释义的完整回答})-
\log p(\text{选择不恰当释义的完整回答}).
\]

- \(S_{\mathrm{context},i}\)：模型看到匹配语境前缀（contextual prefix，完整左文、目标词及其后三个共同词）时的平均回答偏好。
- \(S_{\mathrm{word},i}\)：模型只看到目标词时，对同一候选对已有的平均偏好。
- \(C_i=S_{\mathrm{context},i}-S_{\mathrm{word},i}\)：加入句子语境后产生的偏好变化。

正分数表示更支持恰当候选。正式评分前的 12 个无歧义控制项目全部通过；这只验证显式评分接口正常，不预先决定它是否与生成行为一致。

### 总体关联

\[
\Delta_i=\alpha_0+
\beta_{\mathrm{overall}}S_{\mathrm{context},i}+\varepsilon_i^{(0)}.
\]

\(\beta_{\mathrm{overall}}\) 表示：有语境显式偏好每增加 1 nat，\(\Delta_i\) 平均改变多少。

| 结果 | 数值 |
|---|---:|
| \(\widehat\beta_{\mathrm{overall}}\) | 0.003249 |
| CR1 cluster-robust standard error | 0.000620 |
| 95% confidence interval | [0.002032, 0.004467] |
| 双侧 t test | \(t(552)=5.242\) |
| 双侧 p value | \(2.27\times10^{-7}\) |
| \(R^2\) | 0.0402 |

斜率为正且置信区间完全高于 0，因此显式恰当释义偏好与生成对齐存在显著的正关联。

### 扣除只看目标词偏好后的语境关联

\[
\Delta_i=\alpha_1+
\beta_{\mathrm{context}}C_i+
\beta_{\mathrm{word}}S_{\mathrm{word},i}+
\varepsilon_i^{(1)}.
\]

\(\beta_{\mathrm{context}}\) 的含义是：在两个项目具有相同 word-only score（只看目标词分数）时，语境增量每增加 1 nat，\(\Delta_i\) 平均改变多少。

| 结果 | 数值 |
|---|---:|
| \(\widehat\beta_{\mathrm{context}}\) | 0.002611 |
| CR1 cluster-robust standard error | 0.000676 |
| 95% confidence interval | [0.001283, 0.003939] |
| 双侧 t test | \(t(552)=3.863\) |
| 双侧 p value | \(1.25\times10^{-4}\) |
| \(\widehat\beta_{\mathrm{word}}\) | 0.003730 |
| \(\beta_{\mathrm{word}}\) 95% confidence interval | [0.002396, 0.005065] |
| 模型 \(R^2\) | 0.0462 |

\(\beta_{\mathrm{context}}>0\) 且在统计上显著。准确解释是：**在 \(S_{\mathrm{word}}\) 相同，即回归已经调整测得的只看目标词偏好的项目之间，更大的 \(C_i\) 与更大的 \(\Delta_i\) 相关。** 595 个项目中，479 个的 \(C_i>0\)，116 个的 \(C_i<0\)。

必须区分这个条件关联与 simple correlation（简单相关）。未经调整时，\(C_i\) 与 \(\Delta_i\) 的 Pearson correlation 为 \(r=-0.001\)，几乎为零；同时，\(C_i\) 与 \(S_{\mathrm{word},i}\) 的相关为 \(r=-0.672\)。也就是说，原有 word-only 偏好越大的项目，其加入语境后的增量往往越小，两种成分在简单相关中相互抵消；回归把它们分开后才显现正的语境条件斜率。因此论文不能写“\(C_i\) 本身与 \(\Delta_i\) 正相关”，只能写“控制 \(S_{\mathrm{word},i}\) 后，较大的 \(C_i\) 与较大的 \(\Delta_i\) 相关”。

不过，第二个模型只解释约 4.6% 的项目间差异，比总体模型增加约 0.6 个百分点。显式问答因此能预测生成行为的一小部分变化，但不能代替生成评测。这个结果支持两种行为的部分一致，而不是完全等价。

## 词汇相似度与“纯隐喻效应”

匹配的有语境/只看目标词设计（matched context/word-only design）保持候选、问题、回答格式和评分方法相同。M1同时纳入语境增量与测得的只看目标词偏好；调整后，语境增量的条件斜率仍为正。

这一调整针对测得分数的线性关系。词汇属性与具体语境之间的交互（interaction）仍可能保留；原始材料中的\(I\)也同时包含低语境合理性与错误解释。因此，RQ1反映此\(A/I\)对比下的预测对齐，RQ2反映调整后的项目关联，均不应命名为“纯隐喻解释效应”。

## 运行与文件

正式流程七个阶段全部通过，总耗时 3,935.72 秒，即约 65.6 分钟。其中 continuation generation 为 45.1 分钟，representation/distance 为 13.5 分钟，RQ2 direct judgement 为 4.8 分钟。

- 7,920 个完整且唯一的 item-condition-seed groups
- 253,440 条五-token continuations
- 0 个缺失组，0 个重复组
- 2,640 个 triple-seed distance rows、880 个 triple rows、595 个 target-item rows
- 3,520 个 RQ2 input-order rows、7,040 个 response scores、595 个合并项目

完整结果位于 `runs/formal/20260908T224807Z/`；下载归档为 `transfer/20260908T224807Z-results.tar.gz`。本机与远端 SHA-256 一致：`1489f5dc6d8ddc396e5b1395908211388855d039af2678073fd4b5bb68de096b`。
