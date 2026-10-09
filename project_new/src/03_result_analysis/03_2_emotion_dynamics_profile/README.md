# 第一版情绪动力学画像

读取 `data/llm_analysis/analysis_table.parquet`，生成每位入选用户一行的 `emotion_dynamics_profile_v1.parquet`，以及每位原始用户一行的 `emotion_dynamics_selection_v1.parquet`。输出均位于 `data/llm_analysis/`，不会修改分析数据表。运行环境使用项目已有的 `pandas`、`numpy` 和 `pyarrow`。

从 `project_new` 目录执行：

```powershell
python src/03_result_analysis/03_2_emotion_dynamics_profile/build_profiles.py
```

可以用 `--input`、`--profile-output`、`--audit-output` 指定路径；`--min-posts`、`--min-span-days` 和 `--min-active-days` 可用于敏感性分析。

## 入选条件

默认同时要求：

1. 该用户在分析表中的**每条微博**都有非空 `signed_score`，从而不跨越缺失微博计算相邻转移。
2. 至少 20 条微博，使相邻变化至少有 19 对观测。
3. 首末微博相隔至少 30 天，排除短时间内密集发帖但缺乏纵向跨度的序列。
4. 微博分布在至少 10 个不同的发帖日期，避免只在少数几天密集发帖的序列。

不按情绪极性出现次数筛选用户，以免排除持续积极或持续消极的表达序列。

某种极性从未作为转移起点时，对应的转移概率为空；`n_origin_neg/neu/pos` 给出分母。

选择审计表保存每位用户的帖子数、缺失分数数目、活跃日期数、观察跨度、是否入选及排除原因。

阈值是第一版的操作性选择，不代表研究上的统一标准。

## 指标

按 `user_id`、`created_at`、`post_id` 排序，同秒微博使用微博 ID 决定顺序。分数为 `signed_score`；相邻微博指原顺序中紧邻的两条。

- `mean_score`：分数均值。
- `sd_score`：分数的样本标准差（`ddof=1`）。
- `rmssd_score`：相邻分数差平方的均值再开方，分母为 `n_posts - 1`。
- `p_neg_to_neg` 等九列：由消极、中立、积极三状态构成的 3×3 条件转移概率。对角线分别表示三种极性的惯性；`overall_stay_rate` 是所有相邻转移中极性未变的比例。
- `n_posts`、`n_active_days`、`span_days`、`median_gap_days`、`max_gap_days`：发帖数、不同发帖日期数、首末发布时间跨度及相邻发帖间隔的中位数和最大值，供解释不规则时间序列。概率分母记录在 `n_origin_neg/neu/pos`。

这版画像描述的是**微博情绪表达**的动态特征。

转移指标按发帖顺序计算，没有把不同发帖间隔视为相同的实际时间；比较用户时应同时查看观察跨度和间隔。

恢复时间、情绪多样性及事件类别均不在本版指标中。
