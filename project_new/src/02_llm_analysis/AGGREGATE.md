# 汇总一次 LLM 实验

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

两步应使用同一批清洗数据；汇总程序会核对输入与该实验 `run.json` 中记录的摘要。若本次实验使用了其他 `--input-dir`，建表时也应传入同一个目录。一次只汇总一个实验，不混合不同 run 的模型结果。
