import json
import pymysql
from datetime import datetime

config_file = r"Provincial_Mental_Health_2\weibo_config.json"

with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)


def execute_query(sql, params=None):
    """
    执行SQL查询

    Args:
        sql (str): SQL语句
        params (tuple, optional): 参数. Defaults to None.

    Returns:
        _type_: _description_
    """
    try:
        with pymysql.connect(**config["database"]) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
    except Exception as e:
        print(f"执行SQL语句“{sql}”时发生错误: {e}")
        raise


def execute_many(sql, param_list):
    """
    执行SQL批量操作

    Args:
        sql (str): SQL语句
        param_list (list): 参数列表
    """
    try:
        with pymysql.connect(**config["database"]) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, param_list)
            conn.commit()
    except Exception as e:
        print(f"批量执行SQL语句“{sql}”时发生错误: {e}")
        raise


def get_users(fields=None, where=None, limit=None):
    """
    提取user表数据，返回字典列表

    Args:
        fields (list, optional): 字段列表. Defaults to None.
        where (str, optional): 条件. Defaults to None.
        limit (int, optional): 限制条数. Defaults to None.

    Returns:
        list: 字典列表
    """
    fields = fields or ["id", "screen_name", "gender", "statuses_count", "IP"]
    field_mapping = {"id": "uid"}
    
    sql = f"SELECT {', '.join(fields)} FROM `user`"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"

    result = execute_query(sql)

    users = []
    for row in result:
        user_dict = {}
        for field, value in zip(fields, row):
            mapped_field = field_mapping.get(field, field)
            user_dict[mapped_field] = value
        users.append(user_dict)
    return users


def update_user_done(user_id_list):
    """
    更新user表的done字段为1

    Args:
        user_id_list (list): 用户id列表
    """
    sql = "UPDATE `user` SET `done` = 1 WHERE `id` IN (%s)"
    execute_many(sql, user_id_list)
    print(f"已将 {len(user_id_list)} 个用户设置为已清洗")


def get_weibos(fields=None, where=None, limit=None):
    """
    提取weibo表数据，返回字典列表

    Args:
        fields (list, optional): 字段列表. Defaults to None.
        where (str, optional): 条件. Defaults to None.
        limit (int, optional): 限制条数. Defaults to None.

    Returns:
        list: 字典列表
    """
    fields = fields or ["user_id", "id", "text", "created_at", "attitudes_count", "comments_count", "reposts_count"]

    field_mapping = {"user_id": "uid", "id": "wid"}
    sql = f"SELECT {', '.join(fields)} FROM `weibo`"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"
    result = execute_query(sql)
    weibos = []
    for row in result:
        weibo_dict = {}
        for field, value in zip(fields, row):
            mapped_field = field_mapping.get(field, field)
            weibo_dict[mapped_field] = value

        if "created_at" in weibo_dict and isinstance(weibo_dict["created_at"], datetime):
            weibo_dict["created_at"] = weibo_dict["created_at"].strftime("%Y-%m-%dT%H:%M:%S")
        weibos.append(weibo_dict)
    return weibos


def insert_users_cleaned(user_dicts):
    """
    将清洗后的user数据写入user_cleaned表

    Args:
        user_dicts (list): 字典列表
    """
    if not user_dicts:
        return
    fields = list(user_dicts[0].keys())
    sql = f"INSERT INTO user_cleaned ({', '.join(fields)}) VALUES ({', '.join(['%s']*len(fields))})"
    param_list = [tuple(user[field] for field in fields) for user in user_dicts]
    execute_many(sql, param_list)


def insert_weibos_cleaned(weibo_dicts):
    """
    将清洗后的weibo数据写入weibo_cleaned表

    Args:
        weibo_dicts (list): 字典列表
    """
    if not weibo_dicts:
        return
    fields = list(weibo_dicts[0].keys())
    # 将字符串格式的created_at转换为datetime对象
    for weibo in weibo_dicts:
        if "created_at" in weibo and isinstance(weibo["created_at"], str):
            weibo["created_at"] = datetime.strptime(weibo["created_at"], "%Y-%m-%dT%H:%M:%S")
    sql = f"INSERT INTO weibo_cleaned ({', '.join(fields)}) VALUES ({', '.join(['%s']*len(fields))})"
    param_list = [tuple(weibo[field] for field in fields) for weibo in weibo_dicts]
    execute_many(sql, param_list)








# 示例：仅用于测试
if __name__ == "__main__":
    # users = get_users(limit=5)
    # print(users)
    weibos = get_weibos(limit=5)
    print(weibos)

