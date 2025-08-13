import json
import pymysql
from datetime import datetime
import random
import re
from openai import OpenAI
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict, Any
import logging
from logging.handlers import RotatingFileHandler

import dbPy_analysis as dbPy

# 配置日志系统
def setup_logger():
    # 创建logs目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger('llm_analysis')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器（按大小轮转）
    log_file = os.path.join(log_dir, 'llm_analysis.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()

config_file = r"weibo_config.json"

with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)


def get_post_interval(weibo_list):
    weibo_list = sorted(weibo_list, key=lambda x: x["created_at"])
    weibo_list[0]["post_interval"] = '-'
    weibo_list[-1]["post_interval"] = '-'
    for i in range(len(weibo_list) - 1):
        time1 = weibo_list[i]["created_at"]
        time2 = weibo_list[i+1]["created_at"]
        time_diff = time2 - time1
        post_interval = f"{time_diff.total_seconds() / (24 * 3600):.2f}天"
        weibo_list[i+1]["post_interval"] = post_interval
    return weibo_list


def extract_mentions_and_topics(text):
    # 匹配@用户，以空格结尾
    mentions_pattern = r'@([^\s]+)'
    # 匹配#话题#
    topics_pattern = r'#([^#]+)#'
    
    # 提取@用户
    mentions = re.findall(mentions_pattern, text)
    # 提取话题
    topics = re.findall(topics_pattern, text)
    
    return mentions, topics


RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)


def get_user_weibo_dict(user_ids, user_weibo_list):
    user_weibo_dict = {}
    for user_id in user_ids:
        weibo_list = [weibo for weibo in user_weibo_list if weibo["uid"] == user_id]
        weibo_list = get_post_interval(weibo_list)
        for weibo in weibo_list:
            mentions, topics = extract_mentions_and_topics(weibo["text"])
            weibo["mentions"] = mentions
            weibo["topics"] = topics
        user_weibo_dict[user_id] = weibo_list
    return user_weibo_dict


def build_input_weibo_dict(user_weibo_dict):
    input_weibo_dict = {}
    for user_id, weibo_list in user_weibo_dict.items():
        input_weibo_dict[user_id] = []
        for weibo in weibo_list:
            weibo_dict = {
                "微博ID": weibo["wid"],
                "微博内容": weibo["text"],
                "发布时间": weibo["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            }
            input_weibo_dict[user_id].append(weibo_dict)
    return input_weibo_dict


def save_results_to_json(results, model_name):
    file_path = os.path.join(RESULT_DIR, f"{model_name}_analysis.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def format_llm_output(user_weibo_dict, llm_output):
    # llm_output: {uid: [ {微博ID, 综合情感, 方面级情感, 跨微博关联}, ... ]}
    result = {}
    for uid, weibo_list in user_weibo_dict.items():
        result[uid] = {}
        # 以微博ID为key
        llm_analysis = {item["微博ID"]: item for item in llm_output.get(uid, [])}
        for weibo in weibo_list:
            wid = weibo["wid"]
            entry = {
                "created_at": weibo["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
                "post_interval": weibo.get("post_interval", "-"),
                "mentions": weibo.get("mentions", []),
                "topics": weibo.get("topics", []),
            }
            llm_item = llm_analysis.get(wid, {})
            # 转换综合情感分析结果为英文属性名
            comprehensive_sentiment = llm_item.get("综合情感", {})
            if comprehensive_sentiment:
                entry["comprehensive_sentiment"] = {
                    "type": comprehensive_sentiment.get("类型", ""),
                    "intensity": comprehensive_sentiment.get("强度", 0),
                    "event_summary": comprehensive_sentiment.get("事件概述", ""),
                    "event_category": comprehensive_sentiment.get("事件类型", ""),
                    "keywords": comprehensive_sentiment.get("核心触发词", [])
                }
            else:
                entry["comprehensive_sentiment"] = {}
            
            entry["aspect_sentiment"] = llm_item.get("方面级情感", {})
            entry["narrative_threads"] = llm_item.get("跨微博关联", [])
            result[uid][wid] = entry
    return result


def extract_json_from_markdown(text):
    """使用正则表达式匹配```json和```之间的内容"""
    pattern = r'```json\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # 提取JSON字符串
        json_str = match.group(1)
        try:
            # 将JSON字符串转换为Python字典
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
    else:
        print("未找到JSON内容")
        return None


def sentiment_analysis(llm_config: dict, weibo_list) -> tuple:
    """
    对用户微博进行有效性分析

    Args:
        llm_config (dict): LLM配置
        weibo_list (Any): 用户微博

    Returns:
        str: 分析结果
    """
    base_url = llm_config["base_url"]
    model = llm_config["model"]
    api_key = llm_config["api_key"]
    if isinstance(api_key, list) and len(api_key) == 1:
        api_key = api_key[0]
    if not isinstance(weibo_list, str):
        if isinstance(weibo_list, list):
            weibo_list = json.dumps(weibo_list, ensure_ascii=False)
        else:
            weibo_list = str(weibo_list)
        
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = """
你是一个资深社会心理学家，擅长从社交媒体中识别微妙的情感变化。下面，请基于用户微博数据执行以下分析：
1. 综合情感分析：
   - 识别用户微博中的情感类型，使用"情感类别（积极/消极/中立）-子类（如：喜悦/愤怒/哀愁/迷茫）"的格式进行描述
   - 分析情感的强度（0-10分）
   - 对微博主体事件进行概述，不超过10个字
   - 给出微博的关键词，不超过5个

2. 方面级情感分析（可选）：
   - 识别微博中情感，同样使用"情感类别-子类"的格式进行描述
   - 归纳触发该情感的事件，不超过5个字
   - 以情感为键、事件为值，构建一个字典。若没有方面级情感，则返回空字典

3. 跨微博关联分析（可选）：
   - 定位核心叙事线索（如：日本旅行日记）
   - 检测情绪传染现象（如：股票市场讨论的群体性焦虑）
   - 若没有跨微博关联，则返回空列表


【输入格式】
[
    {
        "微博ID": "1234567890",
        "微博内容": "#胖东来攻入北京# 逛超市也太开心了吧...",
        "发布时间": "2024-12-08 21:56:35"
    }, 
    {...}, 
    ...
]

【输出要求】
请按下面的实例，按照JSON格式输出。注意，该示例仅用于说明输出格式，在进行分析时，请根据实际情况给出结论：
{
    "analysis": [
        {
            "微博ID": "1234567890",
            "综合情感": {
                "类型": "积极-喜悦", 
                "强度": 7, 
                "事件概述": "胖东来开业，用户前去购物",
                "事件类型": "消费行为",
                "核心触发词": ["胖东来", "逛超市", "开心"]
            }, 
            "方面级情感": {
                "消极-抱怨": "超市拥挤",
                "积极-支持": "商品质量好"
            },
            "跨微博关联": ["2024-07日本游记"]
        }, 
        {...},
        ...
    ]
}
"""

    max_retries = 5
    retry_count = 0
    retry_delay = 5  # 重试间隔（秒）

    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": weibo_list},
                ],
                response_format={'type': 'json_object'} if model != "qwen-max" else None,
                max_tokens=4096, 
                temperature=1.0, 
                stream=False
            )
            print(response.usage)
            output = response.choices[0].message.content
            try:
                result = json.loads(output)
                return result
            except:
                result = extract_json_from_markdown(output)
                if result is None:
                    raise ValueError("无法从输出中提取有效的JSON数据！")
                return result
        except Exception as e:
            error_str = str(e)
            if '400' in error_str:
                # logger.warning(f"检测到高风险内容，跳过该用户")
                # logger.warning(f"错误信息：{error_str}")
                return None
            
            retry_count += 1
            if retry_count >= max_retries:
                # logger.error(f"{model} 模型API调用失败次数超过{max_retries}次，程序退出")
                # sys.exit(1)  # 退出程序
                return None
            # logger.warning(f"{model} 模型API调用失败，正在进行第{retry_count}次重试: {str(e)}")
            # time.sleep(retry_delay)  # 等待一段时间后重试



models = [
    config["LLM_API"]["DeepSeek-v3"],
    config["LLM_API"]["Qwen-Max"]
]


class AsyncLLMClient:
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.client = OpenAI(
            api_key=model_config["api_key"][0] if isinstance(model_config["api_key"], list) else model_config["api_key"],
            base_url=model_config["base_url"]
        )
        self.model_name = model_config["model"]
        self.lock = threading.Lock()
        # 修改结果文件路径逻辑
        self.result_dir = os.path.join(RESULT_DIR, f"{'ds' if 'deepseek' in self.model_name.lower() else 'qw'}_analysis")
        os.makedirs(self.result_dir, exist_ok=True)
        self.results = {}
        self.prompt = """
你是一个资深社会心理学家，擅长从社交媒体中识别微妙的情感变化。下面，请基于用户微博数据执行以下分析：
1. 综合情感分析：
   - 识别用户微博中的情感类型，使用"情感类别（仅从'积极/消极/中立'3者中选取）-子类（如：喜悦/愤怒/哀愁/迷茫）"的格式进行描述
   - 分析情感的强度（0-10分）
   - 对微博主体事件进行概述，不超过10个字
   - 给出微博的关键词，不超过5个

2. 方面级情感分析（可选）：
   - 识别微博中情感，同样使用"情感类别-子类"的格式进行描述
   - 归纳触发该情感的事件，不超过5个字
   - 以情感为键、事件为值，构建一个字典。若没有方面级情感，则返回空字典

3. 跨微博关联分析（可选）：
   - 定位核心叙事线索（如：日本旅行日记）
   - 检测情绪传染现象（如：股票市场讨论的群体性焦虑）
   - 若没有跨微博关联，则返回空列表


【输入格式】
[
    {
        "微博ID": "1234567890",
        "微博内容": "#胖东来攻入北京# 逛超市也太开心了吧...",
        "发布时间": "2024-12-08 21:56:35"
    }, 
    {...}, 
    ...
]

【输出要求】
请按下面的实例，按照JSON格式输出。注意，该示例仅用于说明输出格式，在进行分析时，请根据实际情况给出结论：
{
    "analysis": [
        {
            "微博ID": "1234567890",
            "综合情感": {
                "类型": "积极-喜悦", 
                "强度": 7, 
                "事件概述": "胖东来开业，用户前去购物",
                "事件类型": "消费行为",
                "核心触发词": ["胖东来", "逛超市", "开心"]
            }, 
            "方面级情感": {
                "消极-抱怨": "超市拥挤",
                "积极-支持": "商品质量好"
            },
            "跨微博关联": ["2024-07日本游记"]
        }, 
        {...},
        ...
    ]
}
"""
        logger.info(f"初始化模型客户端：{self.model_name}，结果目录：{self.result_dir}")

    def _get_result_file_path(self, uid: str) -> str:
        """根据用户ID获取对应的结果文件路径"""
        first_digit = uid[0]
        prefix = 'ds' if 'deepseek' in self.model_name.lower() else 'qw'
        return os.path.join(self.result_dir, f"{prefix}_{first_digit}.json")

    def _load_existing_results(self, uid: str):
        """加载指定用户ID对应的结果文件"""
        result_file = self._get_result_file_path(uid)
        try:
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    file_results = json.load(f)
                    if uid in file_results:
                        self.results[uid] = file_results[uid]
        except Exception as e:
            logger.error(f"加载结果文件失败：{result_file}，错误：{str(e)}")

    def save_results(self, uid: str):
        """保存结果到对应的文件"""
        try:
            result_file = self._get_result_file_path(uid)
            # 读取现有文件内容
            existing_data = {}
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # 更新数据
            existing_data[uid] = self.results.get(uid, {})
            
            # 创建临时文件
            temp_file = f"{result_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            # 验证临时文件内容
            with open(temp_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                if saved_data != existing_data:
                    raise ValueError("保存的数据与原始数据不匹配")
            
            # 如果验证通过，替换原文件
            os.replace(temp_file, result_file)
            logger.info(f"保存分析结果成功：{result_file}")
        except Exception as e:
            logger.error(f"保存结果文件失败：{result_file}，错误：{str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def update_results(self, user_weibo_dict: Dict[str, List[Dict]], llm_output: Dict[str, List]):
        """更新结果字典"""
        try:
            formatted_result = format_llm_output(user_weibo_dict, llm_output)
            
            # 验证新数据
            for uid, user_data in formatted_result.items():
                if not isinstance(user_data, dict):
                    raise ValueError(f"用户 {uid} 的数据格式错误")
                for wid, weibo_data in user_data.items():
                    if not isinstance(weibo_data, dict):
                        raise ValueError(f"微博 {wid} 的数据格式错误")
            
            # 按文件分组结果
            file_groups = {}
            for uid, user_data in formatted_result.items():
                result_file = self._get_result_file_path(uid)
                if result_file not in file_groups:
                    file_groups[result_file] = {}
                file_groups[result_file][uid] = user_data
            
            # 批量保存每个文件的结果
            for result_file, file_data in file_groups.items():
                self._batch_save_results(result_file, file_data)
            
            logger.info(f"更新{len(user_weibo_dict)}个用户的分析结果 - 模型：{self.model_name}")
        except Exception as e:
            logger.error(f"更新结果失败：{self.model_name}，错误：{str(e)}")
            raise

    def _batch_save_results(self, result_file: str, file_data: Dict[str, Dict]):
        """批量保存结果到指定文件"""
        try:
            # 读取现有文件内容
            existing_data = {}
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # 更新数据
            existing_data.update(file_data)
            
            # 创建临时文件
            temp_file = f"{result_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            # 验证临时文件内容
            with open(temp_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                if saved_data != existing_data:
                    raise ValueError("保存的数据与原始数据不匹配")
            
            # 如果验证通过，替换原文件
            os.replace(temp_file, result_file)
            logger.info(f"批量保存分析结果成功：{result_file}，包含{len(file_data)}个用户的数据")
        except Exception as e:
            logger.error(f"批量保存结果文件失败：{result_file}，错误：{str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    async def analyze_weibo_batch(self, weibo_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """异步分析一批微博"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_config["model"],
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": json.dumps(weibo_batch, ensure_ascii=False)},
                    ],
                    response_format={'type': 'json_object'} if self.model_config["model"] != "qwen-max" else None,
                    max_tokens=4096,
                    temperature=1.0,
                    stream=False
                )
                output = response.choices[0].message.content
                try:
                    result = json.loads(output)
                except:
                    result = extract_json_from_markdown(output)
                logger.info(f"完成微博批次分析 - 模型：{self.model_name}")
                return result
            except Exception as e:
                error_str = str(e)
                # 检查是否是内容审核错误
                if ('data_inspection_failed' in error_str or 
                    'content_filter' in error_str or 
                    'high risk' in error_str or
                    'Risk' in error_str):
                    logger.warning(f"检测到高风险内容，跳过该批次微博")
                    # 获取该批次微博对应的用户ID
                    user_ids = set()
                    try:
                        user_id = str(dbPy.get_uid_from_wid(weibo_batch[0]["微博ID"]))
                        user_ids.add(user_id)
                    except Exception as uid_error:
                        logger.error(f"获取微博 {weibo_batch[0]['微博ID']} 的用户ID失败：{str(uid_error)}")
                    
                    # 更新用户表的error字段
                    if user_ids:
                        update_sql = f"""
                        UPDATE user_cleaned 
                        SET error = 1 
                        WHERE uid IN ({','.join(user_ids)})
                        """
                        try:
                            dbPy.execute_update(update_sql)
                            logger.info(f"已将用户 {user_ids} 标记为错误状态")
                        except Exception as db_error:
                            logger.error(f"更新用户错误状态失败：{str(db_error)}")
                    
                    return None
                
                logger.error(f"微博批次分析出错 - 模型：{self.model_name}，错误：{error_str}，第{retry_count + 1}次尝试")
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"模型 {self.model_name} 分析失败，已达到最大重试次数")
                    error_weibo_list = [weibo["微博ID"] for weibo in weibo_batch]
                    logger.error(f"出错微博列表：{error_weibo_list}")
                    exit(0)
                    return None
                await asyncio.sleep(5)  # 重试前等待5秒
        
        return None


async def process_user_batch(user_batch: List[str], llm_clients: List[AsyncLLMClient]):
    """处理一批用户数据"""
    logger.info(f"开始处理用户批次：{user_batch}")
    
    # 获取这批用户的微博
    user_weibo_list = dbPy.get_weibos(where=f"uid IN ({','.join(user_batch)})")
    logger.info(f"获取到{len(user_weibo_list)}条微博，来自{len(user_batch)}个用户")
    
    # 处理微博数据
    user_weibo_dict = get_user_weibo_dict(user_batch, user_weibo_list)
    input_weibo_dict = build_input_weibo_dict(user_weibo_dict)
    
    # 创建所有模型的分析任务
    async def process_with_model(client: AsyncLLMClient):
        try:
            llm_output = {}
            # 将用户分成2-3组进行并行处理
            user_groups = []
            users = list(input_weibo_dict.items())
            group_size = max(1, len(users) // 9)  # 至少分成4组
            for i in range(0, len(users), group_size):
                user_groups.append(users[i:i + group_size])
            
            logger.info(f"模型 {client.model_name} 将{len(users)}个用户分成{len(user_groups)}组进行并行处理")
            
            # 使用线程池并行处理每组用户
            with ThreadPoolExecutor(max_workers=len(user_groups)) as executor:
                async def process_user_group(user_group):
                    group_output = {}
                    for user_idx, (uid, weibo_input) in enumerate(user_group, 1):
                        # 分批处理，每批最多20条微博
                        batch_size = 20
                        batches = [weibo_input[i:i + batch_size] for i in range(0, len(weibo_input), batch_size)]
                        user_analysis = []
                        for batch_idx, batch in enumerate(batches, 1):
                            logger.info(f"开始分析第{user_idx}/{len(user_group)}个用户的第{batch_idx}/{len(batches)}个微博批次（{len(batch)}条）- 模型：{client.model_name}")
                            result = await client.analyze_weibo_batch(batch)
                            if result and "analysis" in result:
                                user_analysis.extend(result["analysis"])
                                logger.info(f"完成第{user_idx}/{len(user_group)}个用户的第{batch_idx}/{len(batches)}个微博批次分析 - 模型：{client.model_name}")
                            else:
                                logger.warning(f"模型 {client.model_name} 分析第{user_idx}/{len(user_group)}个用户的第{batch_idx}/{len(batches)}个微博批次失败，跳过该批次")
                        if user_analysis:  # 只有在有分析结果时才添加到输出
                            group_output[uid] = user_analysis
                    return group_output
                
                # 创建每个用户组的处理任务
                tasks = [process_user_group(group) for group in user_groups]
                # 等待所有组处理完成
                group_results = await asyncio.gather(*tasks)
                
                # 合并所有组的结果
                for group_output in group_results:
                    llm_output.update(group_output)
            
            if llm_output:  # 只有在有结果时才更新
                client.update_results(user_weibo_dict, llm_output)
                logger.info(f"模型 {client.model_name} 完成当前批次分析")
            else:
                logger.error(f"模型 {client.model_name} 未能成功分析任何批次")
        except Exception as e:
            logger.error(f"模型 {client.model_name} 处理失败：{str(e)}")
            raise
    
    # 并行执行所有模型的分析任务
    try:
        tasks = [process_with_model(client) for client in llm_clients]
        await asyncio.gather(*tasks, return_exceptions=True)  # 添加return_exceptions=True
    except Exception as e:
        logger.error(f"处理用户批次时发生错误：{str(e)}")
        return
    
    # 更新数据库中的done字段
    try:
        # 更新用户表的done字段
        update_user_sql = f"""
        UPDATE user_cleaned 
        SET done = 1 
        WHERE uid IN ({','.join(user_batch)})
        """
        dbPy.execute_update(update_user_sql)
        
        # 更新微博表的done字段
        weibo_ids = [weibo["wid"] for weibo in user_weibo_list]
        if weibo_ids:
            update_weibo_sql = f"""
            UPDATE weibo_cleaned 
            SET done = 1 
            WHERE wid IN ({','.join(map(str, weibo_ids))})
            """
            dbPy.execute_update(update_weibo_sql)
            
        logger.info(f"成功更新用户批次完成状态：{user_batch}")
    except Exception as e:
        logger.error(f"更新完成状态时出错：{str(e)}")


async def main():
    logger.info("开始情感分析处理流程")
    
    # 初始化LLM客户端
    llm_clients = [AsyncLLMClient(model_conf) for model_conf in models]
    logger.info(f"已初始化{len(llm_clients)}个模型客户端")
    
    # 获取所有未处理的用户ID
    sql = """SELECT uid FROM user_cleaned WHERE done = 0 AND error != 1"""
    all_user_ids = dbPy.execute_query(sql)
    all_user_ids = [str(user_id[0]) for user_id in all_user_ids]
    logger.info(f"找到{len(all_user_ids)}个待处理用户")
    
    # 分批处理用户
    batch_size = 20
    for i in range(0, len(all_user_ids), batch_size):
        user_batch = all_user_ids[i:i + batch_size]
        logger.info(f"正在处理第{i+1}到{min(i+batch_size, len(all_user_ids))}个用户，共{len(all_user_ids)}个用户")
        await process_user_batch(user_batch, llm_clients)
    
    logger.info("情感分析处理流程完成")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"程序异常终止：{str(e)}")
        raise