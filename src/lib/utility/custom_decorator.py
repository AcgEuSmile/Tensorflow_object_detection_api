#!/usr/bin/python3
import time

# time estimate decorator
def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = kwargs.get('log_time', method.__name__.upper())
            kwargs['log_time'][name] = te-ts
        else:
            print('{} function spend {} sec'.format(
                method.__name__, te-ts))
        return result

    return timed