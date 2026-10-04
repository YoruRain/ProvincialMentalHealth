# 本地微博清洗

旧版 `cleaning/` 目录保持原样。本目录直接读取 `project_new/data/crawler_results/<uid>/<uid>.json`，不连接数据库。

## 运行

从仓库根目录执行：

```powershell
python project_new/src/02_cleaning/cleaning.py run
python project_new/src/02_cleaning/manual_examine.py
python project_new/src/02_cleaning/cleaning.py export
python project_new/src/02_cleaning/cleaning.py validate-export
```

`run` 每处理一位用户就写入 `project_new/data/cleaning_state/<uid>.json`。再次运行会跳过源文件未变化且已有结果的用户；源文件变化时需显式传入 `run --reprocess`，这会重新判断并覆盖该用户原有的人工决定。`manual_examine.py` 支持 `--uid <用户ID>`，每输入一次 `1` 或 `0` 就立即保存；`s` 跳过，`q` 退出。需要改用其他测试数据时，两个脚本都支持在命令前传入 `--data-root <目录>`。

`export` 从 `auto_accepted` 和 `manual_accepted` 状态生成：

- `project_new/data/cleaning_output/user_cleaned.jsonl`
- `project_new/data/cleaning_output/weibo_cleaned.jsonl`
- `project_new/data/cleaning_output/manifest.json`：记录行数及文件摘要，最后写入。

导出可重复运行，重新生成文件，不会因重复执行而追加重复用户或微博。若导出中断，逐用户状态仍在；重新运行 `export` 即可修复。下游程序读取前可运行 `validate-export` 检查清单。旧 CSV 是历史结果，不会被修改。新输出保留历史 CSV 的业务字段，但不包含分析进度字段 `done`、`error`；`created_at` 保留爬虫的 ISO 字符串。后续分析阶段应使用自己的状态文件，不要改写清洗档案。

## 模型配置

本地配置路径为 `project_new/config/cleaning_models.json`，初始内容为 `{"models": []}`。模型为空时，规则清洗后剩余至少 10 条微博的用户进入人工复核，不会被自动判为通过。需要恢复旧版三模型投票时，填写恰好三个模型：

```json
{
  "models": [
    {"base_url": "", "model": "", "api_key": ""},
    {"base_url": "", "model": "", "api_key": ""},
    {"base_url": "", "model": "", "api_key": ""}
  ]
}
```

填写时替换空字符串；配置不齐时程序会报错。三票同意自动保留，两票同意进入人工复核，其余拒绝。每个模型的最终调用结果写入状态文件的 `model_results`，包括成功投票、明确的内容审核拦截和其他调用错误。审核拦截不计作反对票；用户进入人工复核，`reason` 为 `model_content_filter`。普通调用错误仍标为 `error`，下次 `run` 可重试。状态还会记录模型、时间、本次抽样的微博 ID，以及服务商返回的错误码和请求 ID；不保存 API Key。模型调用需要本地安装 `openai`；空配置和人工复核只使用 Python 标准库。配置文件和数据文件由 Git 忽略，请单独备份。

## 输入约定与状态

每个爬取目录恰有一个 JSON 文件，顶层包含 `user` 对象和 `weibo` 列表，`user.id` 应等于目录名。微博中含 `retweet` 或 `user_id` 与目录不符时跳过并计数；新爬虫应避免产生这类记录。微博去除低价值文本后少于 10 条的用户被拒绝。

`user.ip_location` 取括号前的省份；缺失时从 `project_new/data/collecting/user_id_ip_final.txt` 按 ID 查找。两处都缺失会把用户标为 `error`，后续 `run` 可重试。逐用户状态文件保存源 JSON 的 SHA-256 摘要、清洗结果、模型投票、逐模型调用结果及人工决定；源文件变化不会静默覆盖既有状态。人工复核后仍保留 `model_results`。模型调用进行中会逐模型保存状态；若意外中断，下次 `run` 会重新处理该用户。

仅安排一个清洗或人工复核写进程；不要在导出过程中启动下游读取程序。逐用户状态文件是可恢复的记录，JSONL 是供下游读取的汇总结果。
