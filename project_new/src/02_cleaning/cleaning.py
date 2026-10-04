"""Clean locally crawled Weibo users without a database.

Run ``python cleaning.py run`` to create one resumable state file per user.
Run ``python cleaning.py export`` when downstream JSONL files are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from storage import export_jsonl, load_ip_fallback, load_state, now_utc, save_state, validate_export


LOGGER = logging.getLogger("cleaning")
AD_KEYWORDS = (
    "点开红包", "现金红包", "好礼", "网页链接", "我在参与", "连续签到", "粉打卡",
    "年度歌曲", "免费围观", "关注超话", "蚂蚁庄园：", "森林驿站", "头条文章",
    "注册微博", "闲鱼发布", "闲鱼号", "头像挂件", "试试手气", "个性皮肤",
    "品牌代言", "粉丝头衔", "签到活动", "微博渔场", "添福卡",
)
PROMPT = """你是一名专业的社交媒体信息分析师，请根据用户发布的微博内容判断是否为生活分享账号，需满足以下至少2项特征：
1. 包含具体生活场景/人物互动/个人经历
2. 使用第一人称主观表达（如"我"的感受/经历）
3. 内容呈现非结构化自然叙述（非列表/教程/资讯格式）
4. 涉及日常活动（饮食/出行/家庭/宠物等）

需排除以下特征账号：
• 垂直领域专业内容（医疗/法律/金融等）
• 商品交易/广告推广信息
• 抽象理论/鸡汤语录/政策转载
接下来我将给出一个JSON文件，其中键是用户ID，值是经过采样后的微博内容。
请直接为我返回数字"0"或"1"，"1"代表用户是生活分享账号，"0"代表用户不是生活分享账号。除此之外无需返回其他内容。
"""


def clean_text(text: str) -> str:
    text = re.sub(r"(?:[（(])?分享自(?!己).*$", "", text)
    text = re.sub(r"分享(图片|视频)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid(text: str) -> bool:
    return len(text.strip()) > 1 and not any(keyword in text for keyword in AD_KEYWORDS)


def normalize_region(ip_location: object, fallback: str | None = None) -> str:
    raw = ip_location.strip() if isinstance(ip_location, str) else ""
    raw = re.sub(r"^IP属地\s*[:：]\s*", "", raw)
    region = re.split(r"[（(]", raw, maxsplit=1)[0].strip()
    if region in ("", "未知", "暂无"):
        region = fallback or ""
    if not region:
        raise ValueError("用户缺少 IP 属地，且采集阶段的 ID-属地文件没有回退记录")
    return region


def _sampled_posts(posts: list[dict], max_count: int) -> list[dict]:
    if not posts:
        return []
    step = max(1, len(posts) // max(1, max_count - 1))
    return posts[::step][:max_count]


def sample_weibos(posts: list[dict], max_length: int = 200, max_count: int = 20) -> list[str]:
    return [
        post["text"] if len(post["text"]) <= max_length else post["text"][:max_length] + "..."
        for post in _sampled_posts(posts, max_count)
    ]


def sample_weibo_ids(posts: list[dict], max_count: int = 20) -> list[str]:
    return [str(post["wid"]) for post in _sampled_posts(posts, max_count)]


def _required(mapping: dict, key: str, source: Path):
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{source} 缺少必需字段：{key}")
    return mapping[key]


def _integer(mapping: dict, key: str, source: Path) -> int:
    value = _required(mapping, key, source)
    if isinstance(value, bool):
        raise ValueError(f"{source} 的 {key} 不是整数")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} 的 {key} 不是整数") from error


def read_crawled_user(path: Path, fallback_regions: dict[str, str]) -> dict:
    uid = path.parent.name
    if not uid.isdecimal():
        raise ValueError(f"目录名不是用户 ID：{path.parent}")
    with path.open(encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("user"), dict) or not isinstance(payload.get("weibo"), list):
        raise ValueError(f"{path} 必须包含 user 对象和 weibo 列表")
    source_user = payload["user"]
    if str(_required(source_user, "id", path)) != uid:
        raise ValueError(f"{path} 的 user.id 与目录名不一致")
    user = {
        "uid": uid,
        "screen_name": str(_required(source_user, "screen_name", path)),
        "gender": str(_required(source_user, "gender", path)),
        "statuses_count": _integer(source_user, "statuses_count", path),
        "IP": normalize_region(source_user.get("ip_location"), fallback_regions.get(uid)),
    }
    posts = []
    seen_ids = set()
    skipped_nonoriginal = 0
    for source_post in payload["weibo"]:
        if not isinstance(source_post, dict):
            raise ValueError(f"{path} 包含非对象微博记录")
        # Legacy mock exports contain retweets and other users' posts. Future crawls
        # should contain neither, but keep this guard to avoid a wrong uid mapping.
        if "retweet" in source_post or str(source_post.get("user_id")) != uid:
            skipped_nonoriginal += 1
            continue
        wid = str(_required(source_post, "id", path))
        if not wid.isdecimal() or wid in seen_ids:
            raise ValueError(f"{path} 存在无效或重复微博 ID：{wid}")
        seen_ids.add(wid)
        raw_text = _required(source_post, "text", path)
        if not isinstance(raw_text, str):
            raise ValueError(f"{path} 的微博 {wid} 文本不是字符串")
        text = clean_text(raw_text)
        if not is_valid(text):
            continue
        created_at = str(_required(source_post, "created_at", path))
        try:
            datetime.fromisoformat(created_at)
        except ValueError as error:
            raise ValueError(f"{path} 的微博 {wid} 时间不是 ISO 格式") from error
        posts.append({
            "wid": wid,
            "uid": uid,
            "text": text,
            "created_at": created_at,
            "attitudes_count": _integer(source_post, "attitudes_count", path),
            "comments_count": _integer(source_post, "comments_count", path),
            "reposts_count": _integer(source_post, "reposts_count", path),
        })
    return {"uid": uid, "user": user, "weibos": posts, "skipped_nonoriginal": skipped_nonoriginal}


def load_models(config_path: Path) -> list[dict]:
    if not config_path.exists():
        return []
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    models = config.get("models") if isinstance(config, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"{config_path} 必须包含 models 列表")
    if models and len(models) != 3:
        raise ValueError("模型投票需要配置恰好 3 个模型；空列表表示全部进入人工复核")
    for model in models:
        if not isinstance(model, dict) or not all(isinstance(model.get(key), str) and model[key].strip() for key in ("base_url", "model", "api_key")):
            raise ValueError("每个模型都需要非空的 base_url、model 和 api_key")
    return models


def _provider(base_url: str) -> str:
    host = (urlparse(base_url).hostname or "").lower()
    if "deepseek" in host:
        return "deepseek"
    if "dashscope" in host or "aliyuncs" in host:
        return "qwen"
    if "moonshot" in host or "kimi" in host:
        return "kimi"
    return "unknown"


def _error_details(error: Exception) -> dict:
    details = {
        "http_status": getattr(error, "status_code", None),
        "provider_code": getattr(error, "code", None),
        "request_id": getattr(error, "request_id", None),
        "message": str(error)[:500],
    }
    response = getattr(error, "response", None)
    if response is not None:
        details["http_status"] = details["http_status"] or getattr(response, "status_code", None)
        try:
            payload = response.json()
        except (ValueError, AttributeError):
            payload = None
        if isinstance(payload, dict):
            body = payload.get("error")
            if isinstance(body, dict):
                details["provider_code"] = body.get("code") or body.get("type") or details["provider_code"]
                details["message"] = str(body.get("message") or details["message"])[:500]
            details["request_id"] = payload.get("request_id") or details["request_id"]
    if response is not None and not details["request_id"] and hasattr(response, "headers"):
        details["request_id"] = response.headers.get("x-request-id")
    return details


def _is_content_filter(details: dict) -> bool:
    code = str(details["provider_code"] or "").lower()
    message = details["message"].lower()
    return code in {"data_inspection_failed", "datainspectionfailed", "content_filter"} or (
        "content_filter" in message and "high risk" in message
    )


def vote_with_model(model: dict, uid: str, sample: list[str], sample_wids: list[str]) -> dict:
    from openai import OpenAI  # Only required when models are configured.

    result = {
        "provider": _provider(model["base_url"]), "model": model["model"],
        "sample_wids": sample_wids, "occurred_at": now_utc(),
    }
    client = OpenAI(api_key=model["api_key"], base_url=model["base_url"], max_retries=0)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model["model"],
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": json.dumps({uid: sample}, ensure_ascii=False)},
                ],
                temperature=1.0,
                stream=False,
            )
            result.pop("provider_code", None)
            result.pop("message", None)
            result["attempts"] = attempt + 1
            result["http_status"] = 200
            result["request_id"] = getattr(response, "_request_id", None)
            if response.choices[0].finish_reason == "content_filter":
                result.update(outcome="content_filter", provider_code="content_filter")
                return result
            answer = (response.choices[0].message.content or "").strip()
            if answer not in ("0", "1"):
                raise ValueError(f"模型 {model['model']} 返回无效投票：{answer!r}")
            result.update(outcome="vote", vote=int(answer))
            return result
        except Exception as error:
            details = _error_details(error)
            result.update(details, attempts=attempt + 1)
            if _is_content_filter(details):
                result["outcome"] = "content_filter"
                return result
            if attempt < 4:
                time.sleep(5)
    result["outcome"] = "error"
    return result


def classify_votes(votes: list[int]) -> str:
    if len(votes) != 3 or any(vote not in (0, 1) for vote in votes):
        raise ValueError("分类结果必须是 3 个 0/1 投票")
    score = sum(votes)
    if score == 3:
        return "auto_accepted"
    if score == 2:
        return "needs_review"
    return "rejected"


def _input_files(data_root: Path):
    root = data_root / "crawler_results"
    if not root.is_dir():
        raise FileNotFoundError(f"找不到爬取结果目录：{root}")
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        files = list(directory.iterdir())
        if len(files) != 1 or not files[0].is_file() or files[0].suffix.lower() != ".json":
            raise ValueError(f"{directory} 必须且只能包含一个 JSON 文件")
        yield files[0]


def run_cleaning(data_root: Path, config_path: Path, reprocess: bool = False) -> Counter:
    models = load_models(config_path)
    fallback_regions = load_ip_fallback(data_root)
    counts: Counter = Counter()
    for path in _input_files(data_root):
        uid = path.parent.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = load_state(data_root, uid)
        if previous and previous.get("status") not in ("error", "processing") and not reprocess:
            if previous.get("source_sha256") != digest:
                LOGGER.warning("用户 %s 的源文件已变化；使用 --reprocess 才会覆盖现有状态", uid)
                counts["source_changed"] += 1
            else:
                counts["skipped_existing"] += 1
            continue
        state = {"uid": uid, "source_sha256": digest, "updated_at": now_utc(), "votes": [], "model_results": [], "reason": ""}
        try:
            state.update(read_crawled_user(path, fallback_regions))
            if len(state["weibos"]) < 10:
                state["status"] = "rejected"
                state["reason"] = "too_few_valid_weibos"
            elif not models:
                state["status"] = "needs_review"
                state["reason"] = "models_not_configured"
            else:
                sample = sample_weibos(state["weibos"])
                sample_wids = sample_weibo_ids(state["weibos"])
                state["status"] = "processing"
                for model in models:
                    result = vote_with_model(model, uid, sample, sample_wids)
                    state["model_results"].append(result)
                    if result["outcome"] == "vote":
                        state["votes"].append(result["vote"])
                    state["updated_at"] = now_utc()
                    save_state(data_root, state)
                outcomes = {result["outcome"] for result in state["model_results"]}
                if "content_filter" in outcomes:
                    state["status"] = "needs_review"
                    state["reason"] = "model_content_filter"
                elif "error" in outcomes:
                    state["status"] = "error"
                    state["reason"] = "model_call_failed"
                else:
                    state["status"] = classify_votes(state["votes"])
                    state["reason"] = "model_votes"
        except Exception as error:
            state["status"] = "error"
            state["reason"] = str(error)
            LOGGER.error("用户 %s 清洗失败：%s", uid, error)
        save_state(data_root, state)
        counts[state["status"]] += 1
    return counts


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--config", type=Path, default=project_root / "config" / "cleaning_models.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="清洗爬取结果，并按用户保存状态").add_argument("--reprocess", action="store_true")
    commands.add_parser("export", help="从已接受用户状态重建两份 JSONL")
    commands.add_parser("validate-export", help="核对两份 JSONL 与导出清单")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_root = args.data_root.resolve()
    if args.command == "run":
        counts = run_cleaning(data_root, args.config, reprocess=args.reprocess)
        print("清洗结果：", dict(counts))
        if not load_models(args.config):
            print("未配置模型；符合规则的用户已进入人工复核队列。")
    elif args.command == "export":
        print("导出结果：", export_jsonl(data_root))
    else:
        print("校验通过：", validate_export(data_root))


if __name__ == "__main__":
    main()
