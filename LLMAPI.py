from openai import OpenAI
import json

config_file = r"Provincial_Mental_Health_2\weibo_config.json"
with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)


