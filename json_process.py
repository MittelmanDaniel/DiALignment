import json

def open_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
