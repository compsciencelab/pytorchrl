from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv


class DeceivingRewardsEnv(MiniGridEnv):
    """
    ### Description

    Environment with 2 goals, one easy to get and one requiring more exploration.
    The agent must find the access to a room to get the second reward.

    ### Mission Space

    "get to a green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for the easy goal, and '0.5' for the hard goal.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-DeceivingRewards-v0`
    """

    def __init__(self, **kwargs):

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        size = 20
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):

                if j != 1 and i != 1:
                    continue

                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, yB - 2)
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, 1)

        # Place some lava to make goal more distinguishable
        self.grid.vert_wall(height - 2, room_h + 1, 6, Lava)

        # Place the agent
        self.agent_pos = (width - 5, room_h - 3)
        self.agent_dir = 3

        self.mission = "get to the green goal square"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1
