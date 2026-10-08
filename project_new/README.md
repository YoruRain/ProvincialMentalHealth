# 微博情绪分析：整理版流程

`project_new` 将早期项目中的数据清洗、微博情绪分析和结果分析整理为本地文件流程。代码按阶段放在 `src/`，已有的历史数据放在 `data/datasets/legacy/`。仓库根目录中的旧版代码仍是独立实现；本目录的脚本以 JSON/JSONL 文件衔接，不依赖旧版数据库。

## 目录与数据流

```text
project_new/
├── config/                         # 清洗模型配置、LLM 配置示例
├── data/
│   ├── crawler_results/<uid>/<uid>.json  # 新清洗流程的爬虫输入
│   ├── cleaning_state/<uid>.json          # 每位用户的清洗及人工复核状态
│   ├── cleaning_output/                  # 新清洗导出的 JSONL 与 manifest.json
│   ├── datasets/legacy/                  # 历史清洗数据、历史 LLM 汇总和采集资料
│   └── llm_analysis/                    # LLM 实验及后续分析产物
├── src/
│   ├── 01_cleaning/                # 规则清洗、模型投票、人工复核、导出
│   ├── 02_llm_analysis/            # 分批情绪分析、断点续跑、结果汇总
│   └── 03_result_analysis/
│       ├── 03_1_data_analysis_table/     # 用户、微博、模型结果关联成表
│       └── 03_2_emotion_dynamics_profile/ # 用户情绪动力学画像与筛选审计
└── tests/                          # 各阶段的本地单元测试
```

主要流程为：爬虫 JSON → 清洗状态与人工复核 → 两份清洗后 JSONL → LLM 分批结果 → `llm_analysis.json` → `analysis_table.parquet` → 画像表和筛选审计表。也可以从 `data/datasets/legacy/` 中已有的清洗 JSONL 与历史 `llm_analysis.json` 开始做结果分析，无须重新清洗或调用模型。历史数据是已有资料，不等同于本目录新流程的一次运行产物。

`data/` 下的 JSON、JSONL、CSV、TXT、Parquet，以及 `config/` 下的本地 JSON 配置均被 `.gitignore` 忽略。克隆仓库后不能仅凭代码假定这些输入和结果已存在；请单独保存数据与本地配置。当前 `crawler_results/` 没有爬虫输入，运行新清洗流程前需按上述目录约定提供数据。

## 运行环境

以下命令从**仓库根目录**执行，使用 Python 3.10 或更新版本。纯本地清洗、数据检查和 LLM 结果汇总主要使用标准库；配置模型并调用接口时需要 `openai`。建分析表需要 `pyarrow`，画像步骤需要 `pandas`、`numpy` 和 `pyarrow`。本目录目前没有统一的依赖锁定文件，请在已有研究环境中准备相应包，并保留所用版本记录。

## 路径一：使用历史数据分析结果

已有 `data/datasets/legacy/user_cleaned.jsonl`、`weibo_cleaned.jsonl` 与 `llm_analysis.json` 时，可直接建表和画像；这一路径不调用 LLM：

```powershell
python project_new/src/03_result_analysis/03_1_data_analysis_table/build_table.py
python project_new/src/03_result_analysis/03_2_emotion_dynamics_profile/build_profiles.py
```

建表默认输出 `data/llm_analysis/analysis_table.parquet`，保留没有分析结果的微博，并统计缺失数量。画像默认输出 `emotion_dynamics_profile_v1.parquet`（入选用户）和 `emotion_dynamics_selection_v1.parquet`（全部用户的筛选审计）。默认入选条件包括每条微博有可用分数、至少 20 条微博、跨度至少 30 天且至少 10 个活跃日期。建表参数 `--input-dir`、`--analysis`、`--output` 和画像参数 `--input`、`--profile-output`、`--audit-output` 可改路径。字段、缺失值和筛选规则分别见[分析表说明](src/03_result_analysis/03_1_data_analysis_table/README.md)与[画像说明](src/03_result_analysis/03_2_emotion_dynamics_profile/README.md)。

## 路径二：从爬虫结果重新处理

### 1. 清洗与人工复核

爬虫输入应为 `data/crawler_results/<uid>/<uid>.json`：顶层有 `user` 对象和 `weibo` 列表，用户 ID 与目录名一致。每个用户目录只能有一个 JSON 文件。先按需配置 `config/cleaning_models.json`；未配置模型时，规则清洗后符合条件的用户进入人工复核队列。然后执行：

```powershell
python project_new/src/01_cleaning/cleaning.py run
python project_new/src/01_cleaning/manual_examine.py
python project_new/src/01_cleaning/cleaning.py export
python project_new/src/01_cleaning/cleaning.py validate-export
```

`run` 按用户保存状态；`manual_examine.py` 交互式决定保留或拒绝；`export` 从已接受状态重建 `data/cleaning_output/user_cleaned.jsonl`、`weibo_cleaned.jsonl` 和 `manifest.json`。输入变化不会自动覆盖已有决定；`run --reprocess` 会重新判断该用户并覆盖其人工决定，使用前请核对影响。清洗规则、模型配置、状态字段与导出格式见[清洗说明](src/01_cleaning/README.md)。

### 2. 检查输入并运行 LLM 实验

`analysis.py` 默认读取历史清洗数据。若使用上一步的新导出，`inspect` 和 `run` **都要**指定同一个 `--input-dir`：

```powershell
python project_new/src/02_llm_analysis/analysis.py inspect --input-dir project_new/data/cleaning_output
```

参照 `config/llm_analysis.json.example` 创建本地 `config/llm_analysis.json`，设置配置中 `api_key_env` 指向的环境变量（示例为 `DEEPSEEK_API_KEY`）。确认数据与调用成本后，先用少量用户进行真实接口试运行：

```powershell
python project_new/src/02_llm_analysis/analysis.py run --input-dir project_new/data/cleaning_output --run-id trial-01 --config project_new/config/llm_analysis.json --max-users 2
```

确认试运行结果后，去掉 `--max-users` 并保留相同的 `--run-id` 续跑。每次实验的 `run.json`、`state.jsonl` 和 `results/<uid>/<批次>.json` 保存在 `data/llm_analysis/<run-id>/`；输入摘要、模型、提示词等实验身份变化时须使用新的 run ID。输入 JSONL 只读。接口内容审核拒绝会拆分批次，最终可能排除个别微博；失败批次可在续跑时重试。详细行为见[LLM 分析说明](src/02_llm_analysis/README.md)。

### 3. 汇总并生成结果表

实验覆盖所有目标用户、且没有待重试批次后，才能汇总。仅执行 `--max-users` 的部分试验不能汇总为完整结果：

```powershell
python project_new/src/02_llm_analysis/aggregate.py --run-dir project_new/data/llm_analysis/trial-01
python project_new/src/03_result_analysis/03_1_data_analysis_table/build_table.py --input-dir project_new/data/cleaning_output --analysis project_new/data/llm_analysis/trial-01/llm_analysis.json
python project_new/src/03_result_analysis/03_2_emotion_dynamics_profile/build_profiles.py
```

汇总会验证实验记录、输入与结果，默认将 `llm_analysis.json` 写入该实验目录。建表时必须使用该实验对应的清洗输入；画像脚本默认读取上一步生成的 `analysis_table.parquet`。内容审核排除的微博没有模型结果，仍会出现在分析表中；画像筛选会记录因分数缺失而未入选的用户。详见[汇总说明](src/02_llm_analysis/AGGREGATE.md)。

## 验证与使用边界

```powershell
python -m unittest discover -s project_new/tests -v
```

测试覆盖本地清洗、分批分析与汇总、建表和画像的关键逻辑。单元测试使用构造数据或模拟响应；通过测试不代表已完成真实模型调用，也不代表已对全部历史数据重新运行。清洗状态和 LLM 实验目录应与其对应输入一起保存；历史 JSONL、旧版 Notebook 和新实验结果之间没有自动同步。
