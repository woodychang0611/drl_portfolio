from datetime import datetime
from pandas import Timestamp, DateOffset


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_dict_max(d):
    key = max(d.keys(), key=(lambda k: d[k]))
    return key, d[key]


def find_dict_min(d):
    key = min(d.keys(), key=(lambda k: d[k]))
    return key, d[key]


def offset_date(start_date: Timestamp, value, unit: str):
    supported_units = ['days', 'weeks', 'months', 'years']
    if (unit not in supported_units):
        raise ValueError(f"unit '{unit}' not supported, must be one of {supported_units}")
    kwargs = {unit: value}
    return start_date + DateOffset(**kwargs)
