import json

__all__ = ["readCfg"]

def readCfg(path):
    try:
        with open(path, 'r') as cfg_fd:
            json_fd = json.load(cfg_fd)
            _pathDelSlash(json_fd)
            return json_fd
    except:
        raise("Config file open failed")

def _pathDelSlash(json_file):
    for i in json_file:
        if isinstance(json_file[i], str):
            if '/' in json_file[i]:
                if json_file[i][-1] == '/':
                    json_file[i] = json_file[i][:-1]
        else:
            pass