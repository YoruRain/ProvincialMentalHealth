import re
import os
from dbPy_analysis import execute_update

def extract_first_error_id(log_file_path):
    """
    从日志文件中提取ERROR信息中的出错微博ID列表的第一个ID
    
    Args:
        log_file_path (str): 日志文件路径
        
    Returns:
        list: 包含每个错误列表第一个ID的列表
    """
    first_ids = []
    
    # 编译正则表达式模式
    error_pattern = re.compile(r'ERROR - 出错微博列表：\[(.*?)\]')
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'ERROR - 出错微博列表：' in line:
                    # 使用正则表达式匹配ID列表
                    match = error_pattern.search(line)
                    if match:
                        # 获取ID列表字符串
                        id_list_str = match.group(1)
                        # 分割ID列表并获取第一个ID
                        first_id = id_list_str.split(',')[0].strip("'")
                        first_ids.append(first_id)
    
    except Exception as e:
        print(f"处理日志文件时出错: {str(e)}")
        return []
    
    return first_ids

def extract_error_users_and_models(log_file_path):
    """
    从日志文件中提取出错用户和对应的模型信息
    
    Args:
        log_file_path (str): 日志文件路径
        
    Returns:
        list: 包含(用户ID, 模型名称)元组的列表
    """
    error_info = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                # 匹配用户标记为错误状态的行
                if '已将用户' in line and '标记为错误状态' in line:
                    user_match = re.search(r"已将用户 \{'(\d+)'\} 标记为错误状态", line)
                    if user_match:
                        user_id = user_match.group(1)
                        # 检查下一行是否包含模型信息
                        if i + 1 < len(lines):
                            next_line = lines[i + 1]
                            if '模型' in next_line and '分析' in next_line and '失败' in next_line:
                                model_match = re.search(r"模型 (\S+) 分析", next_line)
                                if model_match:
                                    model = model_match.group(1)
                                    error_info.append((user_id, model))
                i += 1
    
    except Exception as e:
        print(f"处理日志文件时出错: {str(e)}")
        return []
    
    return error_info

def extract_error_ids():
    # 日志文件路径
    log_file = os.path.join('LLM_Analysis', 'logs', 'llm_analysis.log')
    
    # 提取第一个错误ID
    first_ids = extract_first_error_id(log_file)
    
    # 提取出错用户和模型信息
    error_users_models = extract_error_users_and_models(log_file)
    
    # 打印出错用户和模型信息
    print("\n出错用户和对应模型信息：")
    for user_id, model in error_users_models:
        print(f"用户ID: {user_id}, 模型: {model}")
    
    sql = f"""UPDATE user_cleaned SET done = 0 WHERE uid IN (
    SELECT uid FROM weibo_cleaned WHERE wid IN ({','.join(first_ids)}))
    """
    count = execute_update(sql)
    # 打印结果
    print(f"\n错误ID数量: {len(first_ids)}, 更新记录数: {count}")

if __name__ == "__main__":
    log_file = os.path.join('LLM_Analysis', 'logs', 'llm_analysis0609.log')

    lst = extract_error_users_and_models(log_file)
    print(lst)
    # error_id_ms = [tp[0] for tp in lst if tp[1] == 'moonshot-v1-auto']
    # print(error_id_ms)

    # sql = f"UPDATE user_cleaned SET error = 0 WHERE uid IN ({','.join(error_id_ms)})"

    # count = execute_update(sql)