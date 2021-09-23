import random
from animalai.envs.arena_config import Vector3


def analyze_arena(arena):
    tot_reward = 0
    max_good = 0
    goods = []
    goodmultis = []
    for i in arena.arenas[0].items:
        if i.name in ['GoodGoal', 'GoodGoalBounce']:
            if len(i.sizes) == 0:  # arenas max cannot be computed
                return -1
            max_good = max(i.sizes[0].x, max_good)
            goods.append(i.sizes[0].x)
        if i.name in ['GoodGoalMulti', 'GoodGoalMultiBounce']:
            if len(i.sizes) == 0:  # arenas max cannot be computed
                return -1
            tot_reward += i.sizes[0].x
            goodmultis.append(i.sizes[0].x)

    tot_reward += max_good
    goods.sort()
    goodmultis.sort()
    return tot_reward


def random_size_reward():
    # according to docs it's 0.5-5
    s = random.randint(5, 50) / 10
    return (s, s, s)


def set_reward_arena(arena, force_new_size=False):
    tot_reward = 0
    max_good = 0
    goods = []
    goodmultis = []
    for i in arena.arenas[0].items:
        if i.name in ['GoodGoal', 'GoodGoalBounce']:
            if len(i.sizes) == 0 or force_new_size:
                x, y, z = random_size_reward()
                i.sizes = []  # remove previous size if there
                i.sizes.append(Vector3(x, y, z))
            max_good = max(i.sizes[0].x, max_good)
            goods.append(i.sizes[0].x)
        if i.name in ['GoodGoalMulti', 'GoodGoalMultiBounce']:
            if len(i.sizes) == 0 or force_new_size:
                x, y, z = random_size_reward()
                i.sizes = []  # remove previous size if there
                i.sizes.append(Vector3(x, y, z))
            tot_reward += i.sizes[0].x
            goodmultis.append(i.sizes[0].x)

    tot_reward += max_good
    goods.sort()
    goodmultis.sort()
    return tot_reward