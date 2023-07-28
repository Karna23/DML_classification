from argparse import Namespace
import json

def load(config_path = "./config/config.json"):
    with open(config_path, "r") as fp:
        config_obj = json.load(fp, object_hook = lambda x: Namespace(**x))
    return config_obj

# cfg = load("config.json")