from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv, WorldObj, COLORS, fill_coords, point_in_rect


class BlueGoal(WorldObj):
    def __init__(self):
        super().__init__("goal", "blue")
        self.goal_type = "blue_goal"

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class YellowGoal(WorldObj):
    def __init__(self):
        super().__init__("goal", "yellow")
        self.goal_type = "yellow_goal"

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class GreyGoal(WorldObj):
    def __init__(self):
        super().__init__("goal", "grey")
        self.goal_type = "grey_goal"

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class GreenGoal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")
        self.goal_type = "green_goal"

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class MultipleDeceivingRewardsEnv(MiniGridEnv):
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

        size = 40
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    # generate a bunch of rooms

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a room function
        def create_room(upper_right_corner, room_wall_length=8, door_pos=None):

            x, y = upper_right_corner

            self.grid.vert_wall(x, y, room_wall_length)
            self.grid.horz_wall(x, y, room_wall_length)
            self.grid.vert_wall(x + room_wall_length, y, room_wall_length)
            self.grid.horz_wall(x, y + room_wall_length, room_wall_length + 1)

            if door_pos:
                self.grid.set(*door_pos, None)

        # Create rooms
        create_room((4, 4), 8, (4, 8))
        create_room((20, 4), 8, (20, 8))
        create_room((20, 20), 8, (20, 24))

        # Create goals
        # Place a green goal square in the upper-right corner
        self.put_obj(GreenGoal(), width - 2, 1)

        # Place a blue goal square in the bottom-right corner
        self.put_obj(BlueGoal(), 8, 8)
        self.put_obj(YellowGoal(), 24, 8)
        self.put_obj(GreyGoal(), 24, 24)

        # Place the agent
        self.agent_pos = (width - 5, 7)
        self.agent_dir = 3

        self.mission = "get to the green goal square"

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal" and fwd_cell.goal_type == "yellow_goal":
                terminated = True
                reward = self._reward()

            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1
