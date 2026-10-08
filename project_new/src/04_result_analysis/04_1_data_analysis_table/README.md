# 统一分析数据表

从 `project_new/data/datasets/legacy/` 的两份清洗后 JSONL 和 `llm_analysis.json` 生成一行一条微博的 Parquet 表。默认输出为 `project_new/data/llm_analysis/analysis_table.parquet`，不会修改输入文件。运行环境需要 `pyarrow`。

```powershell
python project_new/src/04_result_analysis/04_1_data_analysis_table/build_table.py
```

可用 `--input-dir`、`--analysis`、`--output` 指定其他路径。脚本核对用户 ID、微博 ID 及分析结果的关联，完成后输出微博总数、有分析结果的微博数和缺失数。输出先写入临时文件，成功后替换目标文件。

## 字段约定

- `user_id`、`post_id` 为字符串；`created_at` 为秒精度的 Parquet 时间戳；`province` 来自用户 `IP`，`gender` 保留源值。
- `sentiment_category_raw` 和 `sentiment_subcategory_raw` 是综合情感 `type` 按第一个 `-` 分开的两部分。无子类时子类及 `emotion` 为空；综合情感为空时情感相关字段均为空。
- `valence` 将“积极”映射为 `1`，“消极”映射为 `-1`，“中立”和“中性”映射为 `0`；“混合”“复杂”及其他未知类别为空。`signed_score = valence × intensity`，任一输入为空则为空。`emotion` 保留情感子类；原始类别和子类也分别保留。
- `event_category_raw` 保留模型事件类别，源值缺失或空串时为空；`event_category` 目前对所有行为空。`event_summary` 保留模型事件描述。
- 每位用户按 `created_at`、`post_id` 排序。`delta_t` 是距紧邻上一条微博的**秒数**；`prev_score` 是紧邻上一条的 `signed_score`；`delta_score` 是当前分数减去上一条分数。首条微博的这三个字段为空；若任一分数缺失，相关差值也为空。相同发布时间时按微博 ID 字符串排序。
- `likes`、`comments`、`reposts` 是清洗微博中的计数，转为整数。`has_llm_analysis` 标记该微博是否有分析记录；没有分析结果的微博仍保留在表中。
