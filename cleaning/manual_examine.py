import json
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dbPy
import cleaning


def get_users_to_examine(examine_file: str, done_file: str):
    with open(examine_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        users_to_examine = [line.strip() for line in lines]

    with open(done_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        users_done = [line.strip() for line in lines]

    users_to_examine = [user for user in users_to_examine if user not in users_done]
    print(f"已获取检查用户，共 {len(users_to_examine)} 个，已检查 {len(users_done)} 个")
    return users_to_examine


def get_user_dict(user_id_list: list):

    def get_id_ip_dict(file_path: str, file_name: str) -> dict:
        with open(os.path.join(file_path, file_name), "r", encoding="utf-8") as f:
            id_ip_list = f.readlines()
        id_ip_dict = {line.split(",")[0]: line.split(",")[1].strip() for line in id_ip_list}
        return id_ip_dict
    
    WHERE = f"id IN ({', '.join(user_id_list)}) AND to_be_cleaned = 0"
    users = dbPy.get_users(where=WHERE)
    WHERE = f"user_id IN ({', '.join(user_id_list)}) AND to_be_cleaned = 0"
    weibos = dbPy.get_weibos(where=WHERE)
    id_ip_dict = get_id_ip_dict(file_path=r"collecting", file_name="user_id_ip.txt")
    user_dict = {}
    for user in users:
        user_id = user["uid"]
        user["weibo"] = [weibo for weibo in weibos if weibo["uid"] == user_id]
        if not user["IP"]:
            user["IP"] = id_ip_dict[user_id]
        user_dict[user_id] = user
    print(f"已获取用户字典，共 {len(user_dict)} 个")
    return user_dict


def simple_clean(user_dict: dict):
    users_to_remove = []
    for user_id, user_info in user_dict.items():
        weibo_list = user_info["weibo"]
        for weibo in weibo_list:
            weibo["text"] = cleaning.clean_text(weibo["text"])
        
        weibo_list = [weibo for weibo in weibo_list if cleaning.is_valid(weibo["text"])]
        if len(weibo_list) < 10:
            print(f"用户 {user_id} 的微博数量过少，将被移除: 删除 {len(weibo_list)} 条无效微博，剩余 {len(weibo_list)} 条")
            user_dict.pop(user_id)
            users_to_remove.append(user_id)
            continue
        user_info["weibo"] = weibo_list
    
    print(f"已完成用户微博内容清洗")
    return user_dict, users_to_remove





if __name__ == "__main__":
    examine_file = "./cleaning/users_to_examine.txt"
    done_file = "./cleaning/examine_done.txt"
    user_list = get_users_to_examine(examine_file, done_file)
    user_dict = get_user_dict(user_list)
    user_dict, users_to_remove = simple_clean(user_dict)
    # dbPy.update_user_done(users_to_remove)
    dbPy.update_user_to_be_cleaned(users_to_remove)
    sampled_weibo_dict = cleaning.sample_weibos(user_dict, max_weibo_len=100)
    valid_user_dict = {}
    user_done_list = []
    count = 0
    start_time = time.time()
    for user_id, weibo_list in sampled_weibo_dict.items():
        print('*'*25, f"用户 {user_id} 的微博", '*'*25)
        for index, weibo in enumerate(weibo_list):
            print(f"{index+1}. {weibo}")
            # print()
        valid = input(f"({count + 1}) 是否有效(1/0)：")
        if valid == "1":
            valid_user_dict.update({user_id: user_dict[user_id]})
        count += 1
        user_done_list.append(user_id)
        if count % 10 == 0:
            end_time = time.time()
            duration = end_time - start_time
            cleaning.insert_cleaned_users_weibos(valid_user_dict)
            cleaning.write_file("cleaning", "examine_done.txt", user_done_list)
            valid_user_dict = {}
            user_done_list = []
            print(f"已处理 {count} 个用户，用时 {duration:.2f} 秒，速度 {duration / 10:.2f} 秒/用户。已将相关内容写入文件与数据库中")
            go_on = input("是否继续？退出输入q：")
            if go_on == "q":
                break
            start_time = time.time()
    # print(valid_users)
    # print(len(user_list))

