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

## 数据格式

以下描述当前代码运行后生成的格式。`cleaning_state/<uid>.json` 是每位用户一份完整 JSON；两份 `*_cleaned.jsonl` 是每行一个 JSON 对象。微博 ID 和用户 ID 均保存为字符串，计数字段在新生成的数据中为整数。当前 `project_new/data/cleaning_output/` 中从历史数据迁移的 JSONL 可能将计数保存为字符串，且未必有 `manifest.json`；不要把它们当作执行本程序 `export` 后的示例。

### `project_new/data/cleaning_state/<uid>.json`

文件名中的 `<uid>` 与顶层 `uid`、`user.uid` 以及每条 `weibos[].uid` 对应。顶层字段如下：

| 字段 | 类型与含义 |
| --- | --- |
| `uid` | 字符串；用户 ID。 |
| `source_sha256` | 字符串；原始爬虫 JSON 文件字节的 SHA-256，用于判断输入是否变化。 |
| `updated_at` | 字符串；最后写入状态的 UTC ISO 时间。 |
| `status` | 字符串；清洗或人工复核状态，取值见下表。 |
| `reason` | 字符串；当前状态的原因。异常时可能是错误信息。 |
| `votes` | 整数数组；仅包含成功返回的 `0`/`1` 投票。模型未配置或尚未调用时为 `[]`，模型拦截不算作 `0` 票。 |
| `model_results` | 对象数组；每个已调用模型一条最终结果；未调用模型时为 `[]`。 |
| `user` | 对象；清洗后的用户字段，与下方 `user_cleaned.jsonl` 相同。读取爬虫文件失败时可能不存在。 |
| `weibos` | 对象数组；该用户清洗后保留的微博，字段与下方 `weibo_cleaned.jsonl` 相同。读取爬虫文件失败时可能不存在。 |
| `skipped_nonoriginal` | 整数；因带有 `retweet` 或 `user_id` 不匹配而跳过的微博数；读取爬虫文件失败时可能不存在。 |
| `reviewed_at` | 字符串；人工复核决定的 UTC ISO 时间，仅复核后出现。 |

| `status` | 含义 |
| --- | --- |
| `processing` | 模型调用进行中；每次模型返回后保存一次。中断后下次 `run` 从头处理该用户。 |
| `auto_accepted` | 三个模型都投 `1`，自动保留。 |
| `needs_review` | 未配置模型、两票同意，或至少一个模型发生内容审核拦截，待人工复核；具体原因见 `reason`。 |
| `rejected` | 有效微博不足 10 条，或完整投票后同意票不超过一票。 |
| `error` | 输入处理或模型调用失败；下次 `run` 会重试。此处是顶层清洗状态，不是已移除的用户数据字段 `error`。 |
| `manual_accepted` / `manual_rejected` | 人工决定保留 / 拒绝，附有 `reviewed_at`；`model_results` 保留原调用结果。 |

常见 `reason` 为 `too_few_valid_weibos`、`models_not_configured`、`model_votes`、`model_content_filter`、`model_call_failed` 和 `manual_review`。`model_results[]` 的公共字段为 `provider`（从 API 地址识别的 `deepseek`、`qwen`、`kimi` 或 `unknown`）、`model`、`sample_wids`（此次送检样本的微博 ID 数组）、`occurred_at`（UTC ISO 时间）、`attempts`（本次模型调用尝试次数）和 `outcome`。`outcome` 为 `vote` 时有整数 `vote`；为 `content_filter` 或 `error` 时，按接口实际返回情况记录 `http_status`、`provider_code`、`request_id`、`message`，未提供的值可能为 `null` 或字段缺失。正常响应的 `http_status` 为 `200`。这里保存的是每个模型的最终结果及尝试次数，并非每次重试的独立日志。

例如，Qwen 拦截时 `model_results` 中的一条结果可以是：

```json
{
  "provider": "qwen",
  "model": "qwen-plus",
  "sample_wids": ["900101", "900102"],
  "occurred_at": "2026-10-07T04:00:00+00:00",
  "http_status": 400,
  "provider_code": "data_inspection_failed",
  "request_id": "示例请求ID",
  "message": "Input data may contain inappropriate content.",
  "attempts": 1,
  "outcome": "content_filter"
}
```

下面是一个有效微博不足 10 条的状态文件示例，因此尚无模型结果：

```json
{
  "uid": "123456",
  "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "updated_at": "2026-10-07T04:00:00+00:00",
  "votes": [],
  "model_results": [],
  "reason": "too_few_valid_weibos",
  "user": {"uid": "123456", "screen_name": "示例用户", "gender": "f", "statuses_count": 8, "IP": "江苏"},
  "weibos": [
    {"wid": "900001", "uid": "123456", "text": "今天去公园散步。", "created_at": "2026-04-15T12:00:00", "attitudes_count": 1, "comments_count": 0, "reposts_count": 0}
  ],
  "skipped_nonoriginal": 0,
  "status": "rejected"
}
```

### `project_new/data/cleaning_output/user_cleaned.jsonl`

每行对应一位 `auto_accepted` 或 `manual_accepted` 用户，只包含以下字段。下面仅展示一行：

```jsonl
{"uid":"789012","screen_name":"另一位示例用户","gender":"f","statuses_count":12,"IP":"江苏"}
```

`uid`、`screen_name`、`gender`、`IP` 为字符串；`statuses_count` 为整数。`IP` 是处理后的省份。这里不包含 `done`、`error` 或清洗状态；每位用户在一次导出中至多出现一行。

### `project_new/data/cleaning_output/weibo_cleaned.jsonl`

每行对应一条已保留用户的有效原创微博。下面仅展示其中一行：

```jsonl
{"wid":"900101","uid":"789012","text":"今天去公园散步。","created_at":"2026-04-15T12:00:00","attitudes_count":1,"comments_count":0,"reposts_count":0}
```

`wid`、`uid`、`text`、`created_at` 为字符串；三个互动计数为整数。`created_at` 保留爬虫 JSON 中的 ISO 时间字符串，`uid` 关联 `user_cleaned.jsonl`。这里不包含 `done` 或模型结果；同一 `wid` 在一次导出中至多出现一行。

### `project_new/data/cleaning_output/manifest.json`

`export` 在两份 JSONL 写完后最后写入清单，例如：

```json
{
  "generated_at": "2026-10-07T04:01:00+00:00",
  "users": 1,
  "weibos": 12,
  "user_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "weibo_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
}
```

`generated_at` 是 UTC ISO 时间，`users` 和 `weibos` 是本次导出的行数；两个 SHA-256 字段分别对应完整 `user_cleaned.jsonl` 和 `weibo_cleaned.jsonl` 文件的字节摘要。示例摘要仅用于说明格式。`validate-export` 会核对两份文件与清单中的摘要是否一致；它不检查行数，也不判断导出是否晚于最近一次人工复核。
