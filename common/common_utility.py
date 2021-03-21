from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_dict_max(d):
    key = max(d.keys(), key=(lambda k: d[k]))
    return key,d[key]

def find_dict_min(d):
    key = min(d.keys(), key=(lambda k: d[k]))
    return key,d[key]
