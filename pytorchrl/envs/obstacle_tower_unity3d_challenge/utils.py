import cv2
import itertools
import numpy as np


def box_is_placed(state):
    """
    Given a state returns True if the box is placed on the platform.
    Could be useful for reward scaling.
    """
    box = [132, 180, 170, 230, 78, 255]  # HSV color filters
    frame_HSV = cv2.cvtColor(state, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(
        frame_HSV, (box[0], box[2], box[4]), (box[1], box[3], box[5]))

    return (frame_threshold > 0.1).sum() > 90  # hair can appear purple


def box_location(state):
    """
    Returns a tuple: boolean 'unplaced box is visible' and 'center of it'.
    """
    box = [154, 180, 40, 102, 102, 232]  # HSV color filters
    frame_HSV = cv2.cvtColor(state, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(
        frame_HSV, (box[0], box[2], box[4]), (box[1], box[3], box[5]))
    if (frame_threshold > 0.1).sum() > 20:
        x_ord, y_ord = np.where(frame_threshold > 0.1)
        x_m = np.around(y_ord.mean()).astype(np.uint8)
        y_m = np.round(x_ord.mean()).astype(np.uint8)
        return True, [y_m, x_m]
    else:
        return False, (0, 0)


def place_location(state):
    """
    Returns a tuple: boolean 'if placing platform is visible' and 'center of it'.
    Some lighting issues might occur.
    """
    box = [0, 40, 19, 39, 92, 255]  # HSV color filters
    frame_HSV = cv2.cvtColor(state, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(
        frame_HSV, (box[0], box[2], box[4]), (box[1], box[3], box[5]))
    if (frame_threshold > 0.1).sum() > 10:
        x_ord, y_ord = np.where(frame_threshold > 0.1)
        x_m = np.around(y_ord.mean()).astype(np.uint8)
        y_m = np.round(x_ord.mean()).astype(np.uint8)
        return True, [y_m, x_m]
    else:
        return False, (0, 0)


# all actions
branched_action_space = [3, 3, 2, 3]
possible_vals = [range(_num) for _num in branched_action_space]
all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
action_lookup = {tuple(_action): _scalar for (_scalar, _action) in enumerate(all_actions)}


# reduced actions
reduced_action_lookup_6 = {
    0: [0, 1, 0, 0],  # k
    1: [0, 2, 0, 0],  # l
    2: [1, 0, 0, 0],  # w
    3: [1, 0, 1, 0],  # w + space
    4: [1, 1, 0, 0],  # w + k
    5: [1, 2, 0, 0],  # w + l
}

# reduced actions
reduced_action_lookup_7 = {
    0: [0, 0, 0, 0],
    1: [0, 1, 0, 0],  # k
    2: [0, 2, 0, 0],  # l
    3: [1, 0, 0, 0],  # w
    4: [1, 0, 1, 0],  # w + space
    5: [1, 1, 0, 0],  # w + k
    6: [1, 2, 0, 0],  # w + l
}