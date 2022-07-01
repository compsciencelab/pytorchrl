import cv2
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
action_lookup_6 = {
    (0, 1, 0, 0): 0,  # k
    (0, 2, 0, 0): 1,  # l
    (1, 0, 0, 0): 2,  # w
    (1, 0, 1, 0): 3,  # w + space
    (1, 1, 0, 0): 4,  # w + k
    (1, 2, 0, 0): 5,  # w + l
}

action_lookup_7 = {
    (0, 0, 0, 0): 0,  # nothing
    (0, 1, 0, 0): 1,  # k
    (0, 2, 0, 0): 2,  # l
    (1, 0, 0, 0): 3,  # w
    (1, 0, 1, 0): 4,  # w + space
    (1, 1, 0, 0): 5,  # w + k
    (1, 2, 0, 0): 6,  # w + l
}

action_lookup_8 = {
    (0, 0, 0, 0): 0,  # nothing
    (0, 1, 0, 0): 1,  # k
    (0, 2, 0, 0): 2,  # l
    (1, 0, 0, 0): 3,  # w
    (0, 0, 1, 0): 4,  # space
    (1, 0, 1, 0): 5,  # w + space
    (1, 1, 0, 0): 6,  # w + k
    (1, 2, 0, 0): 7,  # w + l
}

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
    0: [0, 0, 0, 0],  # nothing
    1: [0, 1, 0, 0],  # k
    2: [0, 2, 0, 0],  # l
    3: [1, 0, 0, 0],  # w
    4: [1, 0, 1, 0],  # w + space
    5: [1, 1, 0, 0],  # w + k
    6: [1, 2, 0, 0],  # w + l
}

reduced_action_lookup_8 = {
    0: [0, 0, 0, 0],  # nothing
    1: [0, 1, 0, 0],  # k
    2: [0, 2, 0, 0],  # l
    3: [1, 0, 0, 0],  # w
    4: [0, 0, 1, 0],  # space
    5: [1, 0, 1, 0],  # w + space
    6: [1, 1, 0, 0],  # w + k
    7: [1, 2, 0, 0],  # w + l
}
