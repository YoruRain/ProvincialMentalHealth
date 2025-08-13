import json
import re
import sys
import os
import re
import logging
import logging.config
import time
from datetime import datetime
from openai import OpenAI

# 禁用所有API客户端的HTTP请求日志
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dbPy

# 配置日志系统
def setup_logger():
    """配置日志系统"""
    # 创建logs目录（如果不存在）
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 加载日志配置
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.config")
    logging.config.fileConfig(config_path, encoding='utf-8')
    
    return logging.getLogger("cleaning")

# 创建logger实例
logger = setup_logger()

def write_file(file_path: str, file_name: str, data):
    if len(data) == 0:
        return
    with open(os.path.join(file_path, file_name), "a", encoding="utf-8") as f:
        if isinstance(data, list):
            f.writelines('\n'.join(data))
            f.write('\n')
        else:
            f.write(f"{data}\n")
    print(f"已将内容写入文件 {file_path}/{file_name}")


def is_valid(text: str) -> bool:
    """
    判断该微博是否为有效微博，依据为广告、转载、营销相关的关键词
    如果微博中出现了关键词，则认为是无效微博，直接删除

    Args:
        text (str): 一条微博的文本内容

    Returns:
        bool: 微博是否是否有效
    """
    if len(text.strip()) <= 1:
        logger.debug(f"微博内容过短，判定为无效: {text}")
        return False
    
    # 广告、转载相关关键词
    ad_keywords = [
        '点开红包', '现金红包', '好礼', '网页链接', 
        '我在参与', '连续签到', '粉打卡', '年度歌曲', 
        '免费围观', '关注超话', "蚂蚁庄园：", "森林驿站", 
        "头条文章", "注册微博", "注册微博", "闲鱼发布",
        "闲鱼号", "头像挂件", "试试手气", "个性皮肤", 
        "品牌代言", "粉丝头衔", "签到活动","微博渔场",
        "添福卡"
    ]

    # 微博中含有任意一个关键词，则认为无效
    if any(kw in text for kw in ad_keywords):
        logger.debug(f"微博包含广告关键词，判定为无效: {text}")
        return False
    
    return True


def clean_text(text: str) -> str:
    """
    去除微博文本中的分享来源信息、分享图片/视频、多余空格、纯表情符号

    Args:
        text (str): 一条微博的文本内容

    Returns:
        str: 清洗后的微博文本
    """
    # 去除分享来源信息（直接将"分享自"及其后面的所有信息去除）
    text = re.sub(r'(?:[（(])?分享自(?!己).*$', '', text)

    # 去除"分享图片/视频"文本
    text = re.sub(r'分享(图片|视频)', '', text)
    
    # 去除多余空格
    text = re.sub(r'[\s]+', ' ', text)
    
    # 去除纯表情符号的内容
    if len(re.sub(r'[\s]', '', text)) <= 1:
        return ''
    
    return text.strip()


def get_user_dict(cleaned=True, limit=50) -> dict:
    """
    获取用户字典

    Args:
        cleaned (bool, optional): 是否只获取已经过简单清洗的用户. Defaults to True.
        limit (int, optional): 获取的用户数量. Defaults to 50.

    Returns:
        dict: 用户字典，键为用户的uid，值为用户的基本信息和所有微博
    """

    def get_id_ip_dict(file_path: str, file_name: str) -> dict:
        with open(os.path.join(file_path, file_name), "r", encoding="utf-8") as f:
            id_ip_list = f.readlines()
        id_ip_dict = {line.split(",")[0]: line.split(",")[1].strip() for line in id_ip_list}
        return id_ip_dict

    if cleaned:
        WHERE = "to_be_cleaned != 1 AND done = 0"
    else:
        WHERE = "done = 0"

    LIMIT = limit

    users = dbPy.get_users(where=WHERE, limit=LIMIT)
    if len(users) == 0:
        logging.info("已清洗完毕！")
        sys.exit(1)
    if len(users) < limit:
        logging.info(f"获取用户数为：{len(users)}，即将完成清洗！")
    
    user_dict = {}
    user_id_list = [user["uid"] for user in users]
    logging.info(f"获取用户列表：{user_id_list}")
    weibo_list = dbPy.get_weibos(where=f"user_id in ({', '.join(user_id_list)})")

    id_ip_dict = get_id_ip_dict(file_path=r"collecting", file_name="user_id_ip.txt")

    for user in users:
        user_id = user["uid"]
        user["weibo"] = [weibo for weibo in weibo_list if weibo["uid"] == user_id]

        if not user["IP"]:
            try:
                user["IP"] = id_ip_dict[user_id]
                logger.info(f"已获取用户 {user_id} 的IP地址： {user['IP']}")
            except KeyError:
                user["IP"] = "未知"
                logger.warning(f"未获取到用户 {user_id} 的IP地址")
        user_dict[user_id] = user

    return user_dict


def simple_clean(user_dict: dict):
    """对用户字典进行简单清洗，删除无效微博"""
    users_to_remove = []
    total_weibos_before = 0
    total_weibos_after = 0

    for user_id, user_info in user_dict.items():
        weibo_list = user_info["weibo"]
        total_weibos_before += len(weibo_list)
        
        for weibo in weibo_list:
            weibo["text"] = clean_text(weibo["text"])
        before_len = len(weibo_list)

        weibo_list = [weibo for weibo in weibo_list if is_valid(weibo["text"])]
        after_len = len(weibo_list)
        user_info["weibo"] = weibo_list
        removed_num = before_len - after_len
        total_weibos_after += after_len
        
        if after_len < 10:
            logger.info(f"用户 {user_id} 的微博数量过少，将被移除: 删除 {removed_num} 条无效微博，剩余 {after_len} 条")
            users_to_remove.append(user_id)
        elif removed_num > 0:
            logger.info(f"用户 {user_id} 清洗完成： 删除 {removed_num} 条无效微博，剩余 {after_len} 条")

    for user_id in users_to_remove:
        del user_dict[user_id]
    
    logger.info(f"清洗完成：移除了 {len(users_to_remove)} 个用户")
    return user_dict


def save_data(user_dict: dict, file_path: str, file_name: str):
    """保存用户字典"""
    with open(os.path.join(file_path, file_name), "w", encoding="utf-8") as f:
        json.dump(user_dict, f, ensure_ascii=False, indent=4)


def sample_weibos(user_dict: dict, max_weibo_len: int = 200, max_weibo_count: int = 20) -> dict:
    """
    抽样用户字典中的微博

    Args:
        user_dict (dict): 用户字典
        max_weibo_len (int, optional): 最大微博长度. Defaults to 200.
        max_weibo_count (int, optional): 最大微博数量. Defaults to 20.

    Returns:
        dict: 抽样后的用户字典，键为用户的uid，值为抽样后的微博列表
    """

    def truncate_text(text: str, max_len: int) -> str:
        """截断文本，如果文本长度超过max_len，则截断为max_len，并在末尾添加"..."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    sample_weibo_dict = {}
    for user_id, user_info in user_dict.items():
        weibo_list = user_info["weibo"]
        if len(weibo_list) <= max_weibo_count:
            step = 1
        else:
            step = len(weibo_list) // (max_weibo_count - 1)

        sample_weibo_list = [
            truncate_text(weibo["text"], max_weibo_len)
            for weibo in weibo_list[::step][:max_weibo_count]
        ]
        sample_weibo_dict[user_id] = sample_weibo_list

    return sample_weibo_dict


def validity_analysis(llm_config: dict, user_weibo) -> tuple:
    """
    对用户微博进行有效性分析

    Args:
        llm_config (dict): LLM配置
        user_weibo (Any): 用户微博

    Returns:
        str: 分析结果
    """
    base_url = llm_config["base_url"]
    model = llm_config["model"]
    api_key = llm_config["api_key"]
    if isinstance(api_key, list) and len(api_key) == 1:
        api_key = api_key[0]
    if not isinstance(user_weibo, str):
        if isinstance(user_weibo, dict):
            user_weibo = json.dumps(user_weibo, ensure_ascii=False)
        else:
            user_weibo = str(user_weibo)
        
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = """
你是一名专业的社交媒体信息分析师，请根据用户发布的微博内容判断是否为生活分享账号，需满足以下至少2项特征：
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

    max_retries = 5
    retry_count = 0
    retry_delay = 5  # 重试间隔（秒）

    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_weibo},
                ],
                temperature=1.0, 
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if '400' in error_str:
                logger.warning(f"检测到高风险内容，跳过该用户")
                logger.warning(f"错误信息：{error_str}")
                return None
            
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"{model} 模型API调用失败次数超过{max_retries}次，程序退出")
                sys.exit(1)  # 退出程序
            logger.warning(f"{model} 模型API调用失败，正在进行第{retry_count}次重试: {str(e)}")
            time.sleep(retry_delay)  # 等待一段时间后重试


config_file = r"weibo_config.json"
with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)


def multi_model_validity_analysis(user_weibo_dict: dict) -> dict:
    """使用多个模型对用户微博进行有效性分析

    Args:
        user_weibo_dict (dict): 用户微博字典
        prompt (str): 提示词

    Returns:
        dict: 所有用户的分析结果
    """
    # 存储所有用户的分析结果
    all_results = {}
    total_users = len(user_weibo_dict)
    processed_users = 0
    
    # 定义要使用的三个模型配置
    models = [
        config["LLM_API"]["DeepSeek-v3"],
        config["LLM_API"]["Qwen-Plus"],
        config["LLM_API"]["Moonshot-v1"]
    ]
    
    # 对每个用户进行分析
    for user_id, weibo_list in user_weibo_dict.items():
        user_results = []
        # 构造单个用户的测试数据
        test_data = {user_id: weibo_list}
        
        # 使用三个不同的模型进行分析
        for model_config in models:
            try:
                result = validity_analysis(model_config, test_data)
                
                # 确保结果是 "0" 或 "1"
                if result and result.strip() in ['0', '1']:
                    user_results.append(int(result.strip()))
                else:
                    model = model_config["model"]
                    logger.error(f"模型 {model} 分析用户 {user_id} 时返回异常结果: {result}")
                    user_results.append(None)
            except Exception as e:
                logger.error(f"分析用户 {user_id} 时发生错误: {str(e)}")
                user_results.append(None)
        
        all_results[user_id] = user_results
        logger.info(f"用户 {user_id} 分析完成，分析结果：{user_results}")
        
        processed_users += 1
        logger.info(f"进度: {processed_users}/{total_users} 用户分析完成")
    
    return all_results



def classify_users(results: dict, threshold: int = 2) -> dict:
    users_to_remove = []
    users_to_examine = []
    users_error = []
    # counter = {0:0, 1:0, 2:0, 3:0}

    for user_id, result in results.items():
        if None in result:
            logger.warning(f"分类过程中 用户 {user_id} 出现错误，跳过")
            users_error.append(user_id)
            continue
        
        score = sum(result)
        # counter[score] += 1
        if score < threshold:
            users_to_remove.append(user_id)
        elif threshold <= score < 3:
            users_to_examine.append(user_id)

    return users_to_remove, users_to_examine, users_error


def insert_cleaned_users_weibos(user_dict: dict):
    user_dict_list = list(user_dict.values())
    weibo_dict_list = []
    for user in user_dict_list:
        weibo_dict_list.extend(user.pop("weibo"))
    
    dbPy.insert_users_cleaned(user_dict_list)
    dbPy.insert_weibos_cleaned(weibo_dict_list)


def batch_clean(batch_size: int = 10):
    logger.info(f"开始新的清洗批次，批次大小: {batch_size}")
    
    user_dict = get_user_dict(limit=batch_size)
    logger.info(f"获取到 {len(user_dict)} 个用户数据")

    user_id_list = list(user_dict.keys())


    user_dict = simple_clean(user_dict)
    logger.info(f"简单清洗后剩余 {len(user_dict)} 个用户")

    sample_weibo_dict = sample_weibos(user_dict)
    logger.info("完成微博采样")

    results = multi_model_validity_analysis(sample_weibo_dict)
    logger.info("完成多模型有效性分析")

    users_to_remove, users_to_examine, users_error = classify_users(results)
    logger.info(f"分类结果：需移除 {len(users_to_remove)} 个用户，需人工检查 {len(users_to_examine)} 个用户")

    for user_id in users_to_remove:
        del user_dict[user_id]

    for user_id in users_to_examine:
        del user_dict[user_id]

    for user_id in users_error:
        del user_dict[user_id]

    write_file("./cleaning", "users_to_examine.txt", users_to_examine)
    logger.info(f"已将 {len(users_to_examine)} 个需人工检查的用户ID写入文件")

    write_file("./cleaning", "users_error.txt", users_error)
    logger.info(f"已将 {len(users_error)} 个出现错误的用户ID写入文件")

    logger.info(f"清洗完成：最终有效用户数 {len(user_dict)}")
    insert_cleaned_users_weibos(user_dict)
    logger.info("数据已成功写入清洗后的数据表")

    dbPy.update_user_done(user_id_list)
    logger.info(f"已更新 {len(user_id_list)} 个用户的处理状态")

    return len(user_dict)


def batch_clean_loop(batch_size: int = 20):
    i = 1

    while True:
        start_time = time.time()
        logger.info(f"开始第 {i} 批次清洗：")
        valid_count = batch_clean(batch_size=batch_size)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"第 {i} 批次清洗完成，用时 {duration/60:.2f} 分钟，速度为 {duration/batch_size:.2f} 秒/用户，有效率为 {valid_count/batch_size:.2%}")
        undone = dbPy.get_undone_user_count()
        done = dbPy.get_done_user_count()
        ratio = done / (undone + done)
        logger.info(f"已完成清洗 {done} 个用户，剩余 {undone} 个用户，完成率为 {ratio:.2%}")
        i += 1
        time.sleep(10)


if __name__ == "__main__":
    batch_clean_loop()

