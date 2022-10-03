import random


def dummy_custom_scoring_function(smile):
    return {
        "reward": random.random(),
        "valid_smile": True,
    }

