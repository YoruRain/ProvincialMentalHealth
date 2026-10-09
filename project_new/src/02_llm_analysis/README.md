# 本地 DeepSeek 微博情感分析

此目录将旧版 `LLM_Analysis/llm_analysis.py` 的主要分析流程迁移为本地 JSONL 输入。旧目录保持原样。默认输入为 `project_new/data/datasets/legacy/` 中的历史 `user_cleaned.jsonl` 和 `weibo_cleaned.jsonl`；也可用 `--input-dir` 指向新清洗程序的一次完整导出。输入文件始终只读，`legacy_analysis_status.jsonl` 不参与新实验。

## 使用

从仓库根目录先检查输入，不调用模型：

```powershell
python project_new/src/02_llm_analysis/analysis.py inspect
```

参考 `project_new/config/llm_analysis.json.example` 创建本地 `project_new/config/llm_analysis.json`，在环境变量 `DEEPSEEK_API_KEY` 中设置密钥。真实调用需要安装现有代码使用的 `openai` Python 包。先用少量用户试运行：

```powershell
python project_new/src/02_llm_analysis/analysis.py run --run-id trial-01 --config project_new/config/llm_analysis.json --max-users 2
```

确认结果后，去掉 `--max-users` 并使用**相同的** `--run-id` 续跑。每次 `run` 会扫描、校验输入文件并计算摘要；如果有 `manifest.json`，还会核对清洗导出的摘要。实验目录位于 `project_new/data/llm_analysis/<run-id>/`，包含：

- `run.json`：输入摘要、模型、服务地址、提示词摘要、批次大小和结果格式版本；不存 API Key。这些值变化时必须换一个 run ID。
- `state.jsonl`：每个批次完成后追加事件；同一批次的最后一条事件是当前状态。
- `results/<uid>/<批次编号>.json`：成功批次的格式化结果。原始输出按微博 ID 校验完整后才保存。

程序一次只使用一个模型、一个状态写入者。运行目录有文件锁；成功结果先原子写入，再追加并刷盘状态事件。重启时核对结果文件及其 SHA-256；若上次恰在写完结果后中断，会补记状态，而不重复调用。

模型内容审核拒绝先记在**批次**上，然后把该批次拆为两半继续分析。只有单条微博仍被拒绝时，才把它记为该条微博的 `content_rejected`；其余成功结果保留。用户最终可能是全部成功、完成但存在排除项，或仍有失败/待重试批次。其他接口错误和无效输出最多重试三次，失败记录会在下次运行时重试。状态和普通日志不保存微博正文。

当前实现顺序调用模型；一次完整分析的基础请求量约为 35,642 批，发生重试或审核拆批时还会增加。结果采用每批一个 JSON 文件，便于续跑；旧版 `result_analysis.ipynb` 和 `profile/user_profile.ipynb` 仍读取旧目录，未自动接入本实验结果。

## 汇总实验结果

`aggregate.py` 将一个已完成实验的分批结果汇总为结果分析脚本可读取的 `llm_analysis.json`。它读取实验目录的 `run.json`、`state.jsonl` 和 `results/`，并根据 `run.json` 指向的清洗输入重建批次。清洗后的 JSONL 不需要复制或改写。

从仓库根目录运行：

```powershell
python project_new/src/02_llm_analysis/aggregate.py --run-dir project_new/data/llm_analysis/<run-id>
```

默认输出为该实验目录中的 `llm_analysis.json`。如果实验只处理了部分用户、仍有待重试批次、输入或结果文件摘要不匹配，汇总会失败，且不会替换已有输出。单条微博的内容审核排除不阻止汇总；这些微博不出现在汇总 JSON 中，数量会显示在命令输出中，具体 ID 可从 `state.jsonl` 审计。汇总不会调用模型，也不会修改清洗输入或状态文件。

随后用清洗输出和汇总结果建表：

```powershell
python project_new/src/03_result_analysis/03_1_data_analysis_table/build_table.py --input-dir project_new/data/cleaning_output --analysis project_new/data/llm_analysis/<run-id>/llm_analysis.json
```

两步应使用同一批清洗数据；汇总程序会核对输入与该实验 `run.json` 中记录的摘要。若本次实验使用了其他 `--input-dir`，建表时也应传入同一个目录。一次只汇总一个实验，不混合不同 run 的模型结果。旧版 Notebook 仍需单独适配本实验结果。
