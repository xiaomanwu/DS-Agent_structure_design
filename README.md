# DS Agent

<div align="center">

**用自然语言完成日常数据分析的智能体** · **A natural-language data science agent**

🇨🇳 中文默认展开 · 🇬🇧 English collapsed by default · 点击下方标题栏切换 / click a section header to toggle

</div>

所有结论都由可复现的统计代码算出，不靠大模型"编"数字；没有 LLM 也能完整跑通（默认走确定性路由）。
Every number comes from deterministic, tested code — the LLM is optional and only used to phrase
questions into structured slots.

---

<details open>
<summary><b>🇨🇳 中文</b>（点击收起 / collapse）</summary>

### 能力

| 能力 | 你能这么问 | 背后做了什么 |
| --- | --- | --- |
| **自然语言查数** | 近30天各渠道 GMV 排名 | 语义层把"成交额/销售额/revenue"映射到同一个指标定义，再生成只读 SQL（DuckDB），自动限制行数、拦截写操作；回答里附上知识库的口径提示 |
| **指标异常识别** | 最近有没有异常？GMV 什么时候开始跌的？ | 周季节性基线 + 稳健 z（MAD）+ 双侧 CUSUM：单日尖峰和持续位移都能识别；基线可切 Holt-Winters（加性/乘性）、SARIMA、STL+ETS，或 `auto` 在同一 rolling origin 上回测择优并公布 MASE；**置信度验证卡**（显著性/基线质量/持续性/留一极端值依赖/分段一致性）+ 噪声带图 |
| **异动归因** | 近7天 GMV 为什么跌了？ | 精确加法分解（分段贡献 + 新老客/新增流失），比率指标再拆率效应/结构效应/交互项；**根因下钻决策树**：逐层（最多 3 层）自动选维度×取值路径，输出冰柱图与根因路径排名 |
| **AB 实验设计** | 设计一个支付转化率实验，基线3%，检测2%提升，每天5万流量 | 样本量/MDE/时长/分流方案/护栏指标/上线判据；覆盖 15 种互联网常见实验类型（含集群、switchback、序贯、CUPED、父子、反转、双边等） |
| **常规实验分析** | 看下这个 AB 实验的结果 | SRM 卡方检查、两比例 z / Welch t / 比率指标 delta 法、CUPED、BH 多重比较、分群异质性、上线建议；**置信度验证卡**含置换检验零分布交叉验证图 |
| **时间片轮转实验** | 看下时间片轮转实验的结果 | 以时间片为推断单元（不是市场-日）、HAC 标准误、清洗期、相邻时间片配对差、滞后污染检验 |
| **反转实验** | 做一个反转实验验证这个效应 | 期中两组对调：τ₁+τ₂ 漂移检验、持续性（τ₂ 绝对值 / τ₁ 绝对值）、对称性检验、合并估计 |
| **父子实验** | 父子实验怎么验证两层互不干扰 | 父子交叉表卡方正交性、父层固定效应校正（HC1）、父子交互项、最小格子数保护 |
| **双边实验** | 双边实验，买家侧和卖家侧同时干预 | 2×2（00/10/01/11）四格：两侧直接效应、跨侧溢出比、交互项，市场级 bootstrap |
| **敏感性分析** | 这个实验结论稳健吗，做个敏感性分析 | 缩尾/截尾/去极值/log1p/CUPED/分群固定效应/自助法多口径重估 + A/A 安慰剂假阳性 + 分群刀切 |
| **Uplift 建模** | 用因果森林看哪类用户对补贴最敏感 | 因果树（按效应差分裂）、因果森林（子采样 + honest 评估）、T/S/X 学习器、Qini 曲线、十分位定向收益、交叉拟合 AIPW 校准 |
| **断点回归** | 用断点回归评估会员门槛的效果 | 局部线性（三角核、HC1）、带宽选择与敏感性、密度操纵检验、协变量平衡、安慰剂断点、模糊 RDD（Wald 比） |
| **时序因果** | 这次改版对 GMV 的影响，用时序因果评估 | ARIMA 中断时间序列（对数尺度、趋势 + 星期 + 对照序列、安慰剂干预日、口径曲线）+ 结构时序反事实（局部水平模型、预测区间、干预前一步拟合 R²），并与未校正前后差对照 |
| **因果推断** | 用双重差分评估补贴政策对城市 GMV 的影响 | TWFE DiD（城市+周固定效应）、事件研究与平行趋势联合检验、**PSM**（倾向得分匹配：logit 校准卡尺 + 共同支撑诊断 + 重叠性直方图 + SMD 平衡表）、IPW、AIPW、合成控制（含安慰剂检验）；**置信度验证卡**：DiD 附安慰剂置换（假处理城市重跑）直方图，PSM 汇总平衡/重叠/方法一致性检查，时序因果汇总白噪声/安慰剂/双模型一致性 |
| **开源引擎交叉验证** | 用 DoWhy 开源包交叉验证促销的因果效应 | 同一识别问题交给**现成开源因果推断包**（DoWhy 0.14 backdoor：线性回归/PSM/IPW，可选 EconML 因果森林）重估；跑**安慰剂处理**与**随机共同原因**反驳检验；与自研 PSM/IPW/AIPW 对比相对差——跨引擎一致是结果可信的重要证据；未安装时优雅回退并明确提示 |
| **深度学习异常检测** | 用自编码器/深度学习方法看下 GMV 有没有异常 | **自编码器重建误差**（平稳比率输入 + quiet-window 训练 + 99 分位阈值，PyTorch LSTM-AE 可用时自动启用）+ **孤立森林**多变量评分（水平/环比/同比/趋势比特征）；与经典季节基线交叉验证（Jaccard），置信卡覆盖重建质量/告警率/分数分离度 |
| **统计抽样** | 满意度调查基线70%，检测3个百分点变化，需要多少样本？/ 从用户表分层抽样2000个并检验代表性 | 样本量计算（比例/均值，n = (z₁₋α/₂+z₁₋β)²σ²/δ²，附 z 值、公式与 Cochran 文献 + 效应敏感性表）；简单随机/系统/分层抽样（固定 seed 可复现，分层等比例配额表）；标准误与 95% CI；**代表性检验**：类别列卡方拟合优度 + 份额对照表，数值列 Welch t + KS（均值与分布形状分别检验）；理论见 `docs/methodology.md` §7 |
| **建模能力** | 训练一个预测用户是否转化的分类模型，看特征重要性 | **四段流水线（回归与因果推断共用）**：① 数据预处理（缺失值/NaN 档案与填补、稀疏列与常数列剔除、去重）→ ② 特征工程（时间展开/独热 + **共线性处理**：成对 \|r\|>0.95 与 VIF>10 双重剔除，PSM 倾向得分模型同样先跑这一步）→ ③ 分类与回归（逻辑回归/线性回归 vs 梯度提升自动对比择优，5折CV稳定性）→ ④ **模型验证之后的特征分析**（装了 shap 用真 SHAP 值，否则置换重要性+系数方向并明确标注；因果模型（PSM）之后同样附特征分析图，处理变量入模）+ ROC/混淆矩阵/预测-实际图 + 过拟合缺口/CV稳定性/样本量置信卡 |
| **看板能力** | 给我生成一个 GMV 监控看板 | 一键生成**独立 HTML 监控看板**：KPI 环比卡（GMV/订单/买家/转化率）、趋势+7日均线、渠道/设备拆解、退款率、近期异动表（复用 Agent 同一异常检测器，口径一致）；图表 base64 内嵌，可直接分享/投屏 |
| **方法选型** | 这个场景该用什么方法分析？ | 按场景（有无实验、干预方式、数据结构、目标）在全部方法目录上做决策树排序，给出首选 + 交叉验证备选 + 前提 + 坑 + 稳健性清单 |
| **统计检验** | 用了哪些检验？什么是 Welch t 检验？ | 内置 30+ 项检验的理论卡（H0/公式/前提/判定/文献）；六类分析结论自动附检验表与一致性判读；目录见 `docs/statistical_tests.md` |
| **业务知识问答** | GMV 的口径是什么？和 revenue 有什么区别？ | **LangChain + LangGraph RAG**：`MarkdownHeaderTextSplitter` 按标题切块（`指标口径 > GMV`），自定义 `BaseRetriever`（TF-IDF，标题 2:1 正文）检索，`StateGraph` 走 `retrieve → grade → compose` 流水线；离线确定性、无需向量库与大模型；SQL 回答自动附口径提示，报告自动加业务背景章节 |
| **指标字典管理** | 注册指标：利润率 = (gmv - refund_amount) / gmv | 自定义指标注册前先在数仓上**校验分子分母**，通过即进入语义层——自然语言查询/诊断/归因/报告立刻同口径可用；支持别名、单位、方向、负责人，`指标字典` 查看全部口径，下线保留可追溯；方法论与自查清单见 `docs/metric_dictionary.md`（已入 RAG 语料） |
| **数据源关联** | 关联数据源 /Users/me/sales.csv 叫 sales | 接入**企业数据库**（DuckDB/SQLite/Postgres/MySQL）与**离线表格**（CSV/TSV/Parquet/JSON/Excel），注册进当前会话后自然语言查询、SQL、诊断、建模、看板全部可直接使用；注册清单持久化、重启自动重挂 |
| **报告撰写** | 把上面的分析写成周报 | 结论先行的 Markdown + 自包含 HTML（图表内嵌 base64），中英双语标题，可直接转发 |

追问会继承上下文：先问"GMV 为什么跌"，再问"按品类再拆一下"，指标和窗口都不变。

### 快速开始

```bash
pip install -e .                 # Python >= 3.10
python -m ds_agent.cli build-data   # 生成内置演示数仓（DuckDB）

python -m ds_agent.cli           # 打开网页交互界面（默认，自动开浏览器）
python -m ds_agent.cli serve --port 8000 --no-browser   # 指定端口 / 不开浏览器
python -m ds_agent.cli ask "近7天 GMV 为什么跌了？"      # 单次问答
python -m ds_agent.cli chat      # 终端 REPL（备用）
```

**推荐用网页界面**：KPI 卡片、图表、表格在浏览器里完整渲染，终端只适合单次脚本化问答。

![DS Agent 界面演示](docs/demo.gif)

**演示动图聚焦六大核心场景、中英双语完整两轮**：统计抽样（分层抽样 + 代表性检验）、指标异常与归因、因果推断（PSM + 事后特征分析）、分类模型（预处理 → 特征工程 → 建模 → SHAP）、看板生成、指标定义——先中文逐屏演示，再切英文完整重演（每个英文画面同样是真实运行的英文回答），并包含大模型设置面板；由 `scripts/capture_demo_turns.py` + `scripts/make_demo_gif.py` 生成。

**界面中英双语**：右上角「中文 / EN」一键切换整套界面语言（侧边栏、示例问题、占位提示、欢迎语），偏好记忆在浏览器本地。

**大模型 API（可选）**：右上角 ⚙︎「模型」面板可随时填写任意 OpenAI 兼容端点（Base URL / API Key / 模型名），保存后立即生效并写入 `.env`。配置后大模型承担四件事：**① 意图识别与任务流程编排**（如「先诊断再写报告」会自动串联多个工具）；**② 灵活生成 SQL**（语义层没覆盖的指标，LLM 写 SQL，只读护栏校验通过后才执行）；**③ 灵活生成最终报告**（在工具结果之上生成解读摘要，确定性结论完整保留）；**④ 能力边界判断**（需求超出范围时诚实说明局限并给出最接近的替代问法）。不配置或调用失败时自动回退内置规则路由，全部功能不受影响。也可用环境变量 `DS_AGENT_LLM_BASE_URL` / `_API_KEY` / `_MODEL` 预配置。

**可扩展架构**：新增能力不必改核心代码——实现一个 `Extension`（处理函数 + 路由关键词 + 中英文描述）并 `register_extension(...)` 注册，即自动进入规则路由、大模型规划目录、help 能力清单与网页能力列表（完整示例：`examples/04_custom_extension.py`）。指标定义、数据源接入同样通过注册式 API 挂载，详见 `docs/architecture.md` 的 Extension points。

也可以当库用：

```python
from ds_agent.agent import DSQueryAgent

agent = DSQueryAgent(warehouse="demo_data/warehouse.duckdb")
reply = agent.ask("近7天 GMV 为什么跌了？")
print(reply.text)          # 叙述
print(reply.blocks)        # KPI 卡片 / 图表 / 表格，前端直接渲染
agent.close()
```

### 示例对话

```text
ds> 近7天 GMV 为什么跌了？

**gmv** 在近7天为 23,520,273.31，较上一个7天 下降 1,787,540.15（-7.06%），相对季节性基线 p=0.002，显著。
主因落在 `device` = **iOS**：贡献 -2,734,644.15（占变动 +153%，该分段自身变化 -30.11%）。
下钻一层：在 **iOS** 内部，`is_new_user` = returning 贡献 -1,980,858.46（+72%）。
异常扫描（近90天，周季节性基线 + 稳健 z）：发现 1 段异常区间，最大一段 2026-10-10~2026-10-20
（11天，持续位移），实际 37,640,651.52 vs 预期 40,681,363.23（-7.47%，p=0.0000），严重度 严重。
分段扫描（`device`）：最异常的是 **iOS**，窗口变化 -30.11%（p=0.0025），2026-10-15 偏离 -37.79%（|z|=7.3）。

📌 业务口径提示（来自知识库）：
【指标口径 > GMV（成交总额）】GMV 统计下单时刻的订单金额，包含未支付订单，不扣除退款……
```

```text
ds> 这次改版对 GMV 的影响，用时序因果评估

**时序因果**（干预日 2026-09-06）：ARIMA 中断时间序列给出 +12.04%（375,861.53，p=9.25e-94），
结构时间序列给出 +12.83%（438,793.80，p=2.91e-10）——两个模型一致。
**为什么不信"前后差"**：未校正口径是 +24.12%，比模型高 +11.29%，那部分是本来就在涨的趋势。
**诊断**：ITS 残差 Ljung-Box p=0.237，安慰剂检验 p=0.111；结构模型干预前一步拟合 R²=0.948。

  [ITS 口径敏感性]
  | 设定 | effect | p |
  | 主口径 | +12.04% | 9.2e-94 |
  | 去掉线性趋势 | +11.98% | 7.2e-241 |
  | 去掉星期因子 | +12.08% | 2.2e-98 |
  | 原值尺度季节ARIMA | +15.69% | 2.4e-15 |
```

```text
ds> 这个场景该用什么方法分析？促销是运营手动选人发的

场景解析：observational。首选 **倾向得分匹配（PSM）**：有选择偏差的观察数据——先匹配，
再在匹配样本上重估。
备选：IPW/AIPW（双稳健交叉验证）、敏感性分析。
[方法排序表：方法 / 为什么 / 前提 / 坑]  [稳健性清单：5 项下结论前检查]
```

### 方法要点（详见 `docs/methodology.md`）

- **统计检验**：内置 30+ 项检验目录（正态性 / 方差 / 位置 / 比例 / 分布 / 时序 / 回归 / 重抽样 / 多重检验），每项给出**原假设、统计量、公式、前提、判定规则与文献**；异动诊断、实验读数、DiD、ITS、建模、PSM 六类分析的结论都会自动附「检验表 + 一致性判读」（主检验一致拒绝 / 一致不拒绝 / 不一致时以稳健或非参数检验为准，并明确标注未执行项的原因）。直接问「用了哪些检验」「什么是 Welch t 检验」也能得到理论卡。完整目录与选型指南见 `docs/statistical_tests.md`，每项分析的结论必须携带的六个要素见 `docs/methodology.md` §6。

- **异常识别**：`expected_t = 近28天中位数水平 × 星期几指数`，水平项用 56 天局部增速做前移修正（避免趋势带来的系统性偏差）；残差用 `max(1.4826·MAD, 0.8·内点标准差)` 标准化，单日尖峰与 CUSUM 持续位移分别成段，重复报警用 t 检验（p<0.05）过滤。
- **时序基线**：Holt-Winters（加性/乘性、阻尼）、SARIMA、STL+ETS、季节朴素、EWMA、滚动中位数都在同一 rolling origin 上回测，用 MASE 选型并公布与次优的差距；所有预测只用当天之前的数据，事故不会漏进自己的预期。
- **贡献度**：加法指标严格可加；比率指标拆成 `Σw·Δr + Σr·Δw + ΣΔw·Δr`，三项之和恰好等于总变化（代码里有断言）。
- **根因下钻决策树**：在贡献度拆解之上递归下钻——每个节点在剩余维度里选"最大贡献 × 集中度"最高的维度，按贡献阈值分支 top 分段，最深 3 层；叶子按占总变动比例排名成根因清单，冰柱图里缩进即深度、长度即贡献。
- **置信度验证**：异常识别、实验读数、DiD、PSM、时序因果五类分析都输出标准化检查卡（PASS/WARN/FAIL + 总体置信度），并配图——异常带 ±2.5σ 噪声带，实验附置换检验零分布直方图，DiD 附假处理城市安慰剂分布（固定种子、确定性可复现）。
- **实验**：比例指标用两比例 z，连续指标用 Welch t 且附 bootstrap 区间，比率指标用 delta 法（含分子分母协方差）；多重比较用 BH-FDR；序贯场景提供 O'Brien-Fleming 与 mSPRT。时间片轮转以**时间片**为推断单元并用 HAC 标准误；父子实验做正交性卡方 + 父层固定效应；双边实验从 2×2 四格的边际拆直接效应与跨侧溢出。
- **因果**：DiD 用双向固定效应且只保留交互项；相对效应按处理组反事实 `y10·(y01/y00)` 归一；合成控制用单纯形约束权重 + 空间安慰剂秩检验；**PSM 在 logit 尺度用 0.2·sd 卡尺做 1:1 匹配，报告共同支撑区间与处理组溢出数，重叠性直方图随答案输出**；断点回归用局部线性 + 密度操纵检验 + 协变量平衡；时序因果用对数尺度 ARIMA ITS 与结构时序反事实，两者都要通过安慰剂检验才写进结论。
- **Uplift**：因果树按"效应差"分裂而不是按结果拟合分裂，因果森林用子采样 + honest 评估切分，Qini 曲线与十分位定向收益把 CATE 换成可执行的人群策略。
- **方法选型**：解析场景特征后按决策树在 19 个方法目录上排序，每个推荐都带"为什么/前提/坑"，并附通用稳健性清单（识别假设、安慰剂、口径稳定性、异常个体依赖、效应适用范围）。
- **开源因果引擎**：DoWhy 负责"用别人验证过的代码再算一遍"——`CausalModel` 识别 backdoor 后用线性回归/PSM/IPW 重估，安慰剂处理反驳（伪处理下效应必须≈0）与随机共同原因反驳（加伪混杂后效应必须不动）作为独立置信层；与自研估计的相对差 <20% 记为跨引擎一致。可选 EconML `CausalForestDML` 输出 CATE 分布。未安装时优雅回退到自研路径。
- **深度学习异常检测**：自编码器在**平稳比率输入**（相对 28 天滚动中位数）上训练——原始水平会让重建误差被趋势/星期构成淹没；只在 quiet window（逐日偏离 <5%）上训练，防止模型学会异常本身；阈值取 quiet 窗口误差的 99 分位（MAD 下限兜底）。异常日归因只计 flagged 窗口内真正偏离的天，避免重叠窗口的重复计数。孤立森林在水平/环比/同比/趋势比四特征上隔离异常。与经典季节基线做 Jaccard 交叉验证。装了 PyTorch 自动切换 LSTM-AE 序列路径。
- **业务知识 RAG（LangChain + LangGraph）**：知识库由 LangChain `MarkdownHeaderTextSplitter` 按 `##`/`###` 切块（标题继承父级，`指标口径 > GMV`）；检索是自定义 LangChain `BaseRetriever`——TF-IDF（中文二字组 + 英文词），标题与正文分开归一化打分（2:1）；整条流水线编排成 LangGraph `StateGraph`（`retrieve → grade → compose`），离线、确定性、无需向量库；未装 langchain 时自动回退到等价的纯 Python 路径。SQL 回答与报告自动注入相关口径。

### 配置

复制 `.env.example` 为 `.env`：

| 变量 | 说明 |
| --- | --- |
| `DS_AGENT_WAREHOUSE` | DuckDB 数据文件路径，默认 `demo_data/warehouse.duckdb` |
| `DS_AGENT_ARTIFACT_DIR` | 报告与图表输出目录，默认 `artifacts` |
| `DS_AGENT_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | 任意 OpenAI 兼容端点（OpenAI / DeepSeek / Qwen / vLLM）。不填则完全走规则路由，功能不受影响 |

接自己的数据：最快路径是对 Agent 说「关联数据源 <文件/连接串>」（无需改代码）；要深度定制时在 `ds_agent/semantic.py` 里增加 `Table`（时间列、维度及其别名、指标定义、值标签）。
指标定义只写一次，SQL、异常、归因、报告全部复用它——这是口径一致的关键；临时口径用「指标字典」注册，不必改代码。
业务知识写在 `docs/business_knowledge.md`（按 `##`/`###` 分节即可被 RAG 检索），指标定义知识在 `docs/metric_dictionary.md`，无需改代码。

### 演示数据

| 表 | 行数 | 用途 | 预埋的"真相" |
| --- | --- | --- | --- |
| `orders_daily` | 67,500 | 180 天 × 渠道/设备/地区/品类 日粒度 | 近 11 天 GMV 有一个 -7% 量级的持续位移（feed 渠道埋点中断 + iOS 支付 SDK 回退） |
| `experiment_users` | 100,000 | 用户级 AB 实验 | 转化率真实相对提升 +6% |
| `city_panel` | 800 | 20 城 × 40 周面板 | 第 26 周起 6 城补贴，GMV +9% |
| `promo_users` | 40,000 | 促销曝光观测数据 | 真实 ATE = 12 元，但曝光存在混淆 |
| `switchback_log` | 448 | 16 市场 × 28 天轮转排班 | 处理组 GMV +4.5%，且有星期效应与市场异质性 |
| `reverse_log` | 239,918 | 用户级两期反转实验 | 第一期 +6%、对调后 -4%，持续性约 0.7 |
| `nested_users` | 60,000 | 父子实验（父层 2 臂 × 子层 2 臂） | 子层 +6%，与父层正交、无交互 |
| `two_sided_market` | 240 | 双边市场 2×2（买家/卖家各干预与否） | 买家侧 +8%、卖家侧 +6%、跨侧溢出小、市场级交互为负 |
| `uplift_users` | 24,000 | 用户级补贴实验 + 真实 CATE | ATE 12.5，新客与 feed 渠道效应更大（`true_tau` 存真值） |
| `rdd_scores` | 6,000 | 会员分数与门槛（600 分） | 门槛处结果变量跳跃 +22，协变量连续 |
| `its_series` | 140 | 日粒度 GMV + 两条对照市场序列 | 第 95 天起 +12% 水平位移，底层趋势仍在上升（未校正前后差会读成 +24%） |

### 测试与示例

```bash
pytest -q                              # 308 个测试，覆盖统计正确性与端到端问答
python examples/01_walkthrough.py      # 二十三类能力的完整对话
python examples/02_library_usage.py    # 不经过 agent，直接调用工具函数
python examples/03_core_demo.py        # 核心功能完整演示（含指标字典/数据源/看板/建模）
python examples/04_custom_extension.py # 运行时注册自定义能力扩展
```

### Agent 框架设计

分层流水线架构，核心原则是 **"LLM 可选、确定性优先"**：

```
用户问题（Web / API / CLI）
   │
   ├─► ① 意图路由层    agent.py::route()：规则打分（关键词 + 守卫词 + 能力注册表）
   │                    LLM planner 可选增强（planner.py），失败自动回退规则
   │
   ├─► ② 能力处理器层  20+ 处理器（诊断 / 实验设计与分析 / 因果 / 建模 / 抽样 / 看板 / 指标 / 报告…）
   │                    readouts.py 以 mixin 承载 8 个进阶实验与因果读数
   │
   ├─► ③ 工具层        tools/*：每个分析领域一个确定性模块（输入 DataFrame/参数 → 结果 + 过程日志）
   │
   ├─► ④ 统计原语层    stats/*：功效、检验、CUPED、BH、抽样——所有上层共用，测试锁定
   │
   ├─► ⑤ 数据层        semantic.py 口径唯一来源 + DuckDB 数仓 + 外部数据源关联（datasource.py）
   │
   └─► ⑥ 横切设施      confidence.py 置信卡 · charts.py 图表 · report.py 报告
                        knowledge.py RAG 知识库 · metric_dict.py 指标字典 · registry.py 运行时扩展
```

设计原则：

1. **LLM 可选**：规则路由离线确定性可用；LLM 只做意图编排与结果解读，调用失败自动回退内置规则，全部功能不受影响；
2. **口径唯一**：所有工具引用同一语义层，注册指标先在数仓校验、通过即全链路（查询/诊断/实验/看板）同口径生效；
3. **结论必证**：每类分析自动附检验表（30+ 检验理论卡）与置信度验证卡，方法论透明可审计；
4. **可复现**：同样问题同样答案，抽样带种子，308 个测试锁定统计行为；
5. **可扩展**：registry.py 让新能力模块免改核心代码接入，成为一等路由目标。

### 目录结构

```
ds_agent/
  semantic.py          语义层：指标/维度/值标签，全链路口径唯一来源
  agent.py             路由 + 查询/诊断/实验设计等处理器 + 叙述生成
  readouts.py          8 个进阶实验与因果读数处理器（mixin）
  reply.py             AgentReply 与数字格式化
  confidence.py        置信度验证：五类分析的标准化检查卡 + 安慰剂/置换检验
  knowledge.py         业务知识 RAG：LangChain 切块 + TF-IDF 检索 + LangGraph 流水线（无依赖时回退纯 Python）
  tools/sql.py         只读 SQL 守卫、时间窗解析、指标查询
  tools/anomaly.py     季节性基线、稳健 z、CUSUM、分段扫描（可切时序基线）
  tools/forecast.py    Holt-Winters / SARIMA / STL+ETS / EWMA，滚动回测与 MASE 选型
  tools/contribution.py 加法/比率分解、多维度归因
  tools/root_cause.py  多层下钻决策树根因分析（含值标签、路径排名、冰柱图）
  tools/experiment.py  15 种实验类型、样本量、实验读数
  tools/experiment_designs.py 轮转/反转/父子/双边实验、敏感性分析
  tools/causal.py      DiD、事件研究、PSM（共同支撑+重叠诊断）、IPW、AIPW、合成控制
  tools/causal_oss.py  开源因果引擎适配（DoWhy 反驳检验、EconML 因果森林、跨引擎一致性）
  tools/dl_anomaly.py  深度学习异常检测：自编码器重建误差 + 孤立森林（torch 可选）
  tools/modeling.py    建模能力：特征工程 + 分类/回归（CV 择优）+ 特征分析（SHAP 可选/置换重要性）+ 模型置信卡
  tools/dashboard.py   看板能力：独立 HTML 监控看板（KPI 环比 / 趋势 / 拆解 / 退款率 / 近期异动）
  tools/uplift.py      因果树、因果森林、T/S/X 学习器、Qini
  tools/rdd.py         断点回归（精确/模糊）、带宽、操纵检验
  tools/ts_causal.py   ARIMA 中断时间序列、结构时序反事实
  tools/method_advisor.py 场景解析 + 方法决策树排序
  tools/report.py      Markdown / 自包含 HTML 报告
  stats/               统计原语（功效、检验、CUPED、BH、序贯）
  charts.py            matplotlib → base64 图表（含中文字体回退）
  llm.py               OpenAI 兼容 chat/completions 客户端
  planner.py           LLM 规划器：问题→意图路由、工具结果解读摘要（失败回退规则）
  metric_dict.py       指标字典：自定义指标注册（校验+语义层生效）+ 口径清单 + 下线治理
  tools/datasource.py  数据源关联：DuckDB/SQLite/Postgres/MySQL + CSV/Excel/Parquet/JSON
  registry.py          运行时扩展注册表：新能力模块免改核心代码接入
  web.py + web_static/ FastAPI + 单文件聊天前端（中英双语 + 运行时模型设置面板）
tests/                 308 个测试
examples/              五个可直接运行的脚本
docs/                  架构与方法说明 + statistical_tests.md（检验目录） + business_knowledge.md + metric_dictionary.md（知识库语料）
                       + market_data_agents_comparison.md / _en.md（Data Agent 市场调研与竞品对比，中英双语）
```

### 已知边界

- 语义层是"手写"的：接新表需要补 `Table` 定义，不做自动 schema 推断。
- 归因是贪婪式"剥洋葱"，不是 Shapley 分解；每一步都报告解释了多少变化，便于判断结论是否完整。
- 异常检测按日粒度设计；周/月粒度需调 `period` 与 `trend_window`。
- `auto` 基线是同一 rolling origin 的回测择优，不是概率模型比较：序列短于约 60 个点时，候选之间的差距属于噪声。
- 时间片轮转的推断前提是清洗期足够长；滞后检验是证据，不是证明。
- Uplift 的 Qini 若不做 honest 切分就会偏乐观，所以同时报告 ATE 与交叉拟合 AIPW 的差距作为诊断。
- 断点回归是单 running variable 的局部线性；多分数或"协变量决定阈值"的设计需要另做。
- 合成控制用标准 SLSQP 求权重，供体池很小（<5）时结论不稳。
- PSM 只能平衡观测到的协变量：隐性选择（"老板拍板"）平衡不掉，结论要配合 AIPW 与敏感性分析使用。
- 知识检索是 TF-IDF 而非语义向量：同义改写可能召回不足，把说法写进知识库最稳；检索器挂在 LangChain `BaseRetriever` 接口上，想换向量检索只需替换这一个类。
- SQL 守卫是正则 + 只读连接双保险，不是完备的 SQL 解析器。

</details>

---

<details>
<summary><b>🇬🇧 English</b> (click to expand)</summary>

### What it does

| Capability | Ask like this | Under the hood |
| --- | --- | --- |
| **NL → SQL** | "GMV by channel, last 30 days" | Semantic layer resolves metric aliases, generates read-only DuckDB SQL, caps rows, blocks any write statement; the answer appends the relevant caliber note from the knowledge base |
| **Anomaly detection** | "Anything anomalous lately?" | Weekly-seasonal baseline + robust z (MAD) + two-sided CUSUM: catches both single-day spikes and multi-day level shifts. The baseline can switch to Holt-Winters (additive/multiplicative), SARIMA, STL+ETS, or `auto`, which backtests every candidate on one rolling origin and reports MASE. Ships with a **confidence-validation card** (significance / baseline quality / persistence / leave-one-out outlier dependence / segment agreement) plus a ±2.5σ noise-band chart |
| **Attribution** | "Why did GMV drop in the last 7 days?" | Exact additive decomposition per segment, rate/mix/interaction split for ratio metrics, and a **root-cause drill-down decision tree**: up to 3 levels of automatic dimension × value paths, an icicle chart, and ranked root-cause paths by share of the root move |
| **Experiment design** | "Design a checkout CVR test, 3% baseline, detect 2% lift" | Sample size, MDE, duration, assignment plan, guardrails, ship criteria; 15 internet experiment archetypes |
| **Standard readout** | "Read out this AB test" | SRM chi-square, two-proportion z / Welch t / delta-method ratio test, CUPED, BH-FDR, segment heterogeneity, ship recommendation; the **confidence card** includes a permutation-null histogram cross-check |
| **Switchback** | "Read out the switchback test" | Time block as the inference unit (not market-day), HAC standard errors, washout, paired block differences, lagged-treatment carryover test |
| **Reversal** | "Validate the effect with a reversal test" | Arms swapped mid-flight: tau1+tau2 drift test, persistence ratio, symmetry test, pooled estimate |
| **Parent-child** | "Do my parent and child experiments interfere?" | Chi-square orthogonality, parent fixed-effect adjustment (HC1), parent x child interaction, minimum cell size |
| **Two-sided** | "Two-sided test treating buyers and sellers" | 2x2 cells (00/10/01/11): direct effects per side, cross-side spillover ratio, interaction, market-level bootstrap |
| **Sensitivity analysis** | "Is the result robust?" | Winsorise/trim/drop-top/log1p/CUPED/segment FE/bootstrap specifications, A/A placebo false-positive rate, segment jackknife |
| **Uplift** | "Which users respond most to the promo?" | Causal tree (effect-difference splits), causal forest (subsample + honest evaluation), T/S/X learners, Qini curve, decile targeting, cross-fitted AIPW calibration |
| **RDD** | "Evaluate the membership threshold" | Local linear fits (triangular kernel, HC1), bandwidth selection and sensitivity, density manipulation test, covariate balance, placebo cutoffs, fuzzy Wald ratio |
| **Time-series causal** | "Estimate the launch effect on GMV" | ARIMA interrupted time series (log scale, trend + weekday + controls, placebo dates, specification curve) plus a structural counterfactual with forecast intervals and a pre-period one-step fit, both checked against the naive pre/post reading |
| **Causal inference** | "Did the subsidy policy lift city GMV?" | TWFE DiD, event study with joint pre-trend test, **PSM** (logit caliper matching + common-support diagnostics + overlap histogram + SMD balance table), IPW, AIPW, synthetic control with in-space placebo inference; **confidence card**: DiD adds a fake-treated-cities placebo histogram, PSM aggregates balance/overlap/estimator-agreement checks, time-series causal aggregates whiteness/placebo/model-agreement checks |
| **Open-source cross-validation** | "Cross-validate the promo effect with DoWhy" | The same identification problem is re-estimated with **off-the-shelf causal packages** (DoWhy backdoor: linear/PSM/IPW, optional EconML causal forest), then pushed through **placebo-treatment** and **random-common-cause** refutation tests; the relative gap to the in-house estimators is reported — cross-engine agreement is the evidence that matters. Degrades gracefully when packages are absent |
| **Deep-learning anomaly detection** | "Check GMV anomalies with an autoencoder" | **Autoencoder reconstruction error** (stationary ratio input + quiet-window training + 99th-percentile threshold; a torch LSTM-AE path activates when PyTorch is available) plus **isolation forest** multivariate scoring (level/DoD/WoW/trend-ratio features); cross-checked against the classical seasonal baseline (Jaccard) with a confidence card covering reconstruction quality, alert volume and score separation |
| **Sampling surveys** | "Baseline 70%, detect a 3pp change — how many samples?" / "Draw a stratified sample of 2000 and test representativeness" | Sample-size determination for proportions and means (`n = (z_{1−α/2}+z_{1−β})²σ²/δ²`, z-values, formula and Cochran citation, plus an effect-size sensitivity table); simple-random / systematic / stratified draws with a fixed reproducible seed and proportional allocation table; standard error and 95% CI; **representativeness tests**: chi-square GOF + share table for categorical columns, Welch t + KS for numeric ones (mean and shape checked separately). Theory in `docs/methodology.md` §7 |
| **Modeling** | "Train a classifier to predict conversion and show feature importance" | **Four-stage pipeline shared with causal inference**: 1. data preprocessing (missing/NaN profile and imputation, sparse and constant columns dropped, duplicates removed) → 2. feature engineering (datetime expansion / one-hot + **collinearity handling**: pairwise \|r\|>0.95 then VIF>10 pruning; the PSM propensity model runs the same step) → 3. classification & regression (logistic/linear vs gradient boosting auto-compared, 5-fold CV stability) → 4. **post-validation feature analysis** (true SHAP values when the `shap` package is installed, otherwise permutation importance + coefficient directions, honestly labelled; the causal PSM readout appends the same analysis with the treatment included) + ROC / confusion-matrix / predicted-vs-actual charts + an overfit-gap / CV-stability / sample-size confidence card |
| **Dashboard** | "Generate a GMV monitoring dashboard" | One-click **standalone HTML monitoring view**: KPI cards with WoW deltas (GMV / orders / buyers / conversion), trend + 7-day moving average, channel & device breakdowns, refund rate, recent-anomaly table (reuses the agent's own anomaly detector, so calibers always agree); charts embedded as base64, ready to share or cast |
| **Method selection** | "Which method fits this scenario?" | Parses the scenario (experiment? treatment mechanism? data shape? goal?), ranks the full method catalogue with a decision tree, returns first choice + cross-check candidates + assumptions + pitfalls + a robustness checklist |
| **Statistical tests** | "Which tests do you run?" / "What is a Welch t-test?" | Theory card for each of the 30+ built-in tests (H0/formula/assumptions/decision/citation); six readouts append a test table with an agreement verdict to every conclusion; catalogue in `docs/statistical_tests.md` |
| **Business knowledge Q&A** | "What is the GMV caliber vs revenue?" | **LangChain + LangGraph RAG**: `MarkdownHeaderTextSplitter` chunks by headings (`指标口径 > GMV`), a custom `BaseRetriever` (TF-IDF, title 2:1 body) ranks them, and a `StateGraph` runs the `retrieve → grade → compose` pipeline — offline, deterministic, no vector store; SQL answers embed caliber notes, reports open with a business-context section |
| **Metric dictionary** | "Register metric profit_rate = (gmv - refund_amount) / gmv" | Custom metrics are **validated against the live warehouse** before registration, then activated into the semantic layer — NL querying, diagnosis, attribution and reports use the new caliber immediately; aliases, unit, direction and owner are tracked; `metric dictionary` lists everything, deprecation keeps lineage; methodology + checklist in `docs/metric_dictionary.md` (part of the RAG corpus) |
| **Data sources** | "Attach data source /Users/me/sales.csv as sales" | Connect **enterprise databases** (DuckDB/SQLite/Postgres/MySQL) and **offline spreadsheets** (CSV/TSV/Parquet/JSON/Excel); attached sources are instantly usable from NL querying, guarded SQL, diagnosis, modeling and dashboards; the registry persists and re-attaches on startup |
| **Reporting** | "Turn this into a weekly report" | Conclusion-first Markdown + self-contained HTML (charts embedded as base64), bilingual headings |

Follow-ups keep context: ask "why did GMV drop", then "break it down by category" — same metric,
same window, same session.

### Quickstart

```bash
pip install -e .
python -m ds_agent.cli build-data
python -m ds_agent.cli           # web UI (default) - opens the browser automatically
python -m ds_agent.cli serve --port 8000 --no-browser
python -m ds_agent.cli ask "Why did GMV drop in the last 7 days?"
python -m ds_agent.cli chat      # terminal REPL (fallback)
```

**The web UI is the recommended way to interact**: KPI cards, charts and tables
render fully in the browser; the terminal is only for scripted one-shot asks.

![DS Agent UI demo](docs/demo.gif)

**The demo animation focuses on six core scenarios, played twice — fully in
Chinese, then fully in English**: statistical sampling (stratified draw +
representativeness tests), anomaly & attribution, causal inference (PSM +
post-causal feature analysis), classification model (preprocess → feature
engineering → model → SHAP), dashboard, metric definitions. Every English
frame is a genuine English agent run, not a translation mock. Generated by
`scripts/capture_demo_turns.py` + `scripts/make_demo_gif.py`.

**Bilingual UI**: the 「中文 / EN」 toggle in the top bar switches the entire
interface language (sidebar, sample questions, placeholder, greeting); the
preference is remembered in the browser.

**Optional LLM API**: the ⚙︎ "Model" panel in the top bar accepts any
OpenAI-compatible endpoint (base URL / API key / model). Saving applies it
immediately and persists to `.env`. When configured the LLM does four things:
**① intent recognition + task-workflow orchestration** (e.g. "diagnose it,
then write it up" chains multiple tools automatically); **② flexible SQL
generation** for metrics outside the semantic layer — always re-validated by
the read-only guard before execution; **③ flexible final-report narrative**
on top of verified tool output (the deterministic answer is always kept);
**④ limitation assessment** — when a request is out of scope it says so
honestly and suggests the closest available capability. Without it — or on
any LLM failure — the agent silently falls back to the built-in
deterministic router. Env vars `DS_AGENT_LLM_BASE_URL` / `_API_KEY` /
`_MODEL` work as an alternative.

**Extensible by design**: new capabilities need no core-code changes —
implement an `Extension` (handler + routing keywords + zh/en metadata) and
`register_extension(...)`; it immediately joins the rule router, the LLM
planner catalog, `help` and the web capability list (full example:
`examples/04_custom_extension.py`). Metric definitions and data sources mount
through the same registration-style APIs — see *Extension points* in
`docs/architecture.md`.

As a library:

```python
from ds_agent.tools.anomaly import AnomalyDetector
from ds_agent.tools.sql import SQLTool, Warehouse

sql = SQLTool(Warehouse("demo_data/warehouse.duckdb"))
series = sql.metric_series("gmv", grain="day")
result = AnomalyDetector().detect(series["date"], series["metric_value"].to_numpy(), label="gmv")

for episode in result["episodes"]:
    print(episode["start"], episode["end"], episode["kind"], f"{episode['impact_pct']:+.1%}")
# 2026-10-10 2026-10-20 level_shift -7.5%
```

### Sample transcript

```text
ds> Why did GMV drop in the last 7 days?

**gmv** over last 7 days is 23,520,273.31, down 1,787,540.15 vs previous 7 days (-7.06%);
vs its seasonal baseline p=0.002 (significant).

Primary driver: `device` = **iOS**, contributing -2,734,644.15 (+153% of the move; its own change -30.11%).
One level deeper: inside **iOS**, `is_new_user` = returning contributes -1,980,858.46 (+72%).
Anomaly scan (90d, weekly-seasonal baseline + robust z): 1 episode, largest 2026-10-10~2026-10-20
(11d, level shift), actual 37,640,651 vs expected 40,681,363 (-7.47%, p=0.0000), severity critical.
Segment scan (`device`): worst is **iOS**, window change -30.11% (p=0.0025), -37.79% on 2026-10-15 (|z|=7.3).

Business caliber (knowledge base): GMV counts orders at placement time, includes unpaid ones,
and does not subtract refunds...
```

```text
ds> Estimate the launch effect on GMV with a time-series causal model

**Time-series causal** (intervention 2026-09-06): ARIMA ITS gives +12.04% (375,861.53, p=9.25e-94),
structural model +12.83% (438,793.80, p=2.91e-10) - the two models agree.
**Why not the pre/post difference**: the naive reading is +24.12%, 11.29pp higher; that gap is the
trend that was already there.
**Diagnostics**: ITS residual Ljung-Box p=0.237, placebo p=0.111; structural pre-period one-step
R2=0.948, placebo p=0.111.

  [ITS specification curve]
  | specification | effect | p |
  | main | +12.04% | 9.2e-94 |
  | no linear trend | +11.98% | 7.2e-241 |
  | no day-of-week terms | +12.08% | 2.2e-98 |
  | seasonal ARIMA, raw scale | +15.69% | 2.4e-15 |
```

### Method notes (full detail in `docs/methodology.md`)

- **Baseline**: `expected_t = median(level over the previous 28 days, deseasonalized) × weekday index`,
  with a 56-day local growth rate added back to remove the lag bias of a trailing window. Residuals are
  scaled by `max(1.4826·MAD, 0.8·inlier std)`; point spikes and CUSUM shifts become separate episodes,
  and a t-test gates the shift episodes (p < 0.05).
- **Time-series baselines**: Holt-Winters (additive/multiplicative, damped), SARIMA, STL+ETS, seasonal
  naive, EWMA and a trailing median all run on the same rolling origin; MASE picks the winner and the
  margin over the runner-up is reported. Every forecast uses only data before the day it predicts, so an
  incident cannot leak into its own expectation.
- **Contribution**: additive metrics decompose exactly; ratio metrics split into
  `Σw·Δr + Σr·Δw + ΣΔw·Δr`, asserted to sum to the total change.
- **Root-cause decision tree**: recursive drill-down on top of the decomposition - each node picks
  the remaining dimension with the best "top contribution × concentration", branches into its top
  segments above a contribution threshold, up to 3 levels; leaves ranked by share of the root move
  form the root-cause shortlist, drawn as an icicle chart (indent = depth, length = contribution).
- **Statistical tests**: a built-in catalogue of 30+ tests (normality / variance / location /
  proportion / distribution / time series / regression / resampling / multiplicity), each carrying
  its **null hypothesis, statistic, formula, assumptions, decision rule and citation**. Six readouts
  (diagnosis, experiment readout, DiD, ITS, modelling, PSM) append a **test table plus an agreement
  verdict** to every conclusion (all primary tests reject / all retain / disagree -> trust the robust
  or non-parametric test), and skipped tests state why. Ask "which tests do you run" or "what is a
  Welch t-test" for a theory card. Full catalogue and a selection guide: `docs/statistical_tests.md`;
  the six elements every conclusion must carry: `docs/methodology.md` §6.
- **Confidence validation**: the five first-class readouts (anomaly, AB, DiD, PSM, time-series
  causal) all end with a standardized check card (PASS/WARN/FAIL + an overall verdict) and a chart -
  a ±2.5σ noise band for anomalies, a permutation-null histogram for AB tests, and a fake-treated-
  cities placebo histogram for DiD (seeded, deterministic, reproducible).
- **Experiments**: two-proportion z for rates, Welch t + bootstrap CI for means, delta method for
  ratio metrics (including the numerator/denominator covariance), BH-FDR across the metric family,
  O'Brien-Fleming and mSPRT for sequential looks. Switchback estimates at the **block** level with HAC
  errors; parent-child tests orthogonality by chi-square and adjusts for parent fixed effects; the
  two-sided design reads direct effects off the 2x2 marginals and reports cross-side spillover as a
  ratio.
- **Causal**: two-way fixed effects DiD with the interaction only; relative effect normalised on the
  treated counterfactual `y10·(y01/y00)`; synthetic control with simplex-constrained weights and an
  in-space placebo rank test; **PSM matches 1:1 on the logit with a 0.2·sd caliper and reports the
  common-support interval, the treated units outside it, and an overlap histogram**; RDD with local
  linear fits, a density manipulation test and covariate balance; time-series causal with a log-scale
  ARIMA ITS and a structural counterfactual, both of which must survive placebo checks.
- **Uplift**: the causal tree splits on the *difference in effect* rather than on outcome fit, the forest
  uses subsampling plus an honest evaluation split, and the Qini curve / decile targeting table turn a
  CATE into an executable targeting rule.
- **Method selection**: the advisor parses scenario features and ranks the 19-method catalogue with a
  decision tree; every recommendation carries why/assumption/pitfall plus the universal robustness
  checklist (identification from design, placebo checks, spec stability, outlier dependence, scope).
- **Open-source causal engines**: DoWhy provides "compute it again with battle-tested code" — a
  `CausalModel` identifies the backdoor and re-estimates via linear regression/PSM/IPW; the
  placebo-treatment refuter (effect must vanish under a fake treatment) and random-common-cause
  refuter (effect must survive a fake confounder) act as an independent confidence layer; a <20%
  relative gap to the in-house estimator counts as cross-engine agreement. EconML `CausalForestDML`
  adds a CATE distribution when installed. Absent packages degrade gracefully to the in-house path.
- **Deep-learning anomaly detection**: the autoencoder trains on **stationary ratio input**
  (deviation from the 28-day trailing median) — raw levels let trend/weekday composition swamp the
  reconstruction error; it trains only on quiet windows (every day within 5% of the trailing median)
  so it cannot learn the anomaly itself; the cut is the 99th percentile of the quiet-error
  distribution (MAD floor). Anomalous-day attribution counts only deviating days inside flagged
  windows, avoiding double-counting from overlapping windows. The isolation forest works on
  level/DoD/WoW/trend-ratio features. A torch LSTM-AE path activates when PyTorch is installed.
- **Business-knowledge RAG (LangChain + LangGraph)**: the KB is chunked by LangChain's
  `MarkdownHeaderTextSplitter` (`##`/`###` headings, titles inherit parents); retrieval is a custom
  LangChain `BaseRetriever` — TF-IDF over Chinese character bigrams + ASCII words, title and body
  scored separately (2:1); the whole flow is orchestrated as a LangGraph `StateGraph`
  (`retrieve → grade → compose`). Offline and deterministic, no vector store; without langchain
  installed the module falls back to an equivalent pure-Python path. Wired into SQL answers and the
  report's business-context section.

### Agent framework design

A layered pipeline whose core principle is **"LLM optional, determinism first"**:

```
User question (Web / API / CLI)
   │
   ├─► ① Intent routing     agent.py::route(): rule scoring (keywords + guard words + capability registry)
   │                         optional LLM planner (planner.py); falls back to rules on failure
   │
   ├─► ② Capability handlers  20+ handlers (diagnosis / experiment design & readouts / causal /
   │                            modelling / sampling / dashboard / metric dictionary / reporting…)
   │                            readouts.py carries 8 advanced experiment & causal readouts as a mixin
   │
   ├─► ③ Tool layer         tools/*: one deterministic module per analysis domain
   │                         (DataFrame/params in → results + process log out)
   │
   ├─► ④ Statistical primitives  stats/*: power, tests, CUPED, BH, sampling — shared by all layers,
   │                              locked by tests
   │
   ├─► ⑤ Data layer         semantic.py (single source of calibers) + DuckDB warehouse
   │                         + external source registration (datasource.py)
   │
   └─► ⑥ Cross-cutting      confidence.py confidence cards · charts.py charts · report.py reports
                             knowledge.py RAG · metric_dict.py metric dictionary · registry.py runtime extensions
```

Design principles:

1. **LLM optional**: the rule router works fully offline and deterministically; the LLM only orchestrates
   intent and narrates results, and any failure falls back to built-in rules with zero feature loss;
2. **One caliber**: every tool reads the same semantic layer; a registered metric is warehouse-validated,
   then live across querying, diagnosis, experiments and dashboards;
3. **Evidence attached**: every analysis auto-attaches its test table (30+ theory cards) and confidence
   card — methodology is transparent and auditable;
4. **Reproducible**: same question, same answer; seeded sampling; 308 tests lock statistical behavior;
5. **Extensible**: registry.py admits new capability modules as first-class routing targets without
   touching core code.

### Configuration

| Variable | Meaning |
| --- | --- |
| `DS_AGENT_WAREHOUSE` | DuckDB file path (default `demo_data/warehouse.duckdb`) |
| `DS_AGENT_ARTIFACT_DIR` | Where reports and charts are written (default `artifacts`) |
| `DS_AGENT_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | Any OpenAI-compatible endpoint. Leave empty to run fully offline on the deterministic router |

To plug in your own warehouse, add a `Table` to `ds_agent/semantic.py` (time column, dimensions and
their aliases, metric definitions, value labels). Define each metric once; SQL, anomaly detection,
attribution and reports all reuse it. Put business knowledge into `docs/business_knowledge.md`
(one `##`/`###` section per topic) and the RAG picks it up with no code changes.

### Tests and examples

```bash
pytest -q                             # 308 tests
python examples/01_walkthrough.py     # every capability, end to end
python examples/02_library_usage.py   # tools called directly, no agent loop
python examples/03_core_demo.py       # full demo of core features (metric dict, sources, dashboard, modeling)
python examples/04_custom_extension.py # register a custom capability at runtime
```

### Known limits

- The semantic layer is hand-written; there is no schema autodiscovery.
- Attribution is greedy peeling, not Shapley: every step reports how much of the move it explains.
- Anomaly detection is tuned for daily data; weekly/monthly series need `period` / `trend_window` changes.
- `auto` baseline selection is a rolling-origin backtest, not a probabilistic model comparison: with
  fewer than ~60 observations the margin between candidates is noise.
- Switchback inference assumes the washout is long enough; the carryover test is evidence, not proof.
- Uplift Qini numbers are in-sample optimistic unless the forest's honest split is respected — the
  reported ATE/AIPW gap is the diagnostic for that.
- RDD is local linear with a single running variable; multiple scores or a covariate-dependent cutoff
  need a different design.
- Synthetic control is unreliable with a donor pool smaller than ~5 units.
- PSM balances only observed covariates: hidden selection survives, so pair it with AIPW and a
  sensitivity pass before concluding.
- Knowledge retrieval is TF-IDF, not semantic embeddings: paraphrases may under-recall — phrase the
  wording into the knowledge base for reliability. The retriever sits behind the LangChain
  `BaseRetriever` interface, so swapping in a vector retriever means replacing one class.
- The SQL guard is a regex plus a read-only connection, not a full SQL parser.

</details>
