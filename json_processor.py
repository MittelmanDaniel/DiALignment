import json

def json_dict(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_json(my_dict):
    with open("data.json", "w") as json_file:
        json.dump(my_dict, json_file, indent=4)
