import ray
import time
import base64
import lz4.frame
from ray import cloudpickle as pickle

FREE_DELAY_S = 10.0
MAX_FREE_QUEUE_SIZE = 100
_last_free_time = 0.0
_to_free = []


def ray_get_and_free(object_ids):
    """
    Call ray.get and then queue the object ids for deletion.
    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.

    Adapted from https://github.com/ray-project/ray/blob/master/rllib/utils/memory.py

    Parameters
    ----------
    object_ids : ObjectID|List[ObjectID]
        Object ids to fetch and free.

    Returns
    -------
    result : python objects
        The result of ray.get(object_ids).
    """

    global _last_free_time
    global _to_free

    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _to_free.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_to_free) > MAX_FREE_QUEUE_SIZE
        or now - _last_free_time > FREE_DELAY_S):
        ray.internal.free(_to_free)
        _to_free = []
        _last_free_time = now

    return result

def broadcast_message(key, message):
    ray.worker.global_worker.redis_client.set(key, message)

def check_message(key):
    return ray.worker.global_worker.redis_client.get(key)

def average_gradients(grads_list):
    """
    Averages gradients coming from distributed workers.

    Parameters
    ----------
    grads_list : list of lists of tensors
        List of actor gradients from different workers.
    Returns
    -------
    avg_grads : list of tensors
        Averaged actor gradients.
    """
    avg_grads = [
        sum(d[grad] for d in grads_list) / len(grads_list) if
        grads_list[0][grad] is not None else 0.0
        for grad in range(len(grads_list[0]))]
    return avg_grads


def pack(data):
    """ from https://github.com/ray-project/ray/blob/master/rllib/utils/compression.py """

    data = pickle.dumps(data)
    data = lz4.frame.compress(data)
    # TODO(ekl) we shouldn't need to base64 encode this data, but this
    # seems to not survive a transfer through the object store if we don't.
    data = base64.b64encode(data).decode("ascii")
    return data


def unpack(data):
    """ from https://github.com/ray-project/ray/blob/master/rllib/utils/compression.py """
    data = base64.b64decode(data)
    data = lz4.frame.decompress(data)
    data = pickle.loads(data)
    return data


