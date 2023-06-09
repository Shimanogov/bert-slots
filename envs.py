import gym
import numpy as np
import skimage
from gym.utils import seeding
from gym import spaces
import copy
from matplotlib import pyplot as plt


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def diamond(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width // 2, r0 + width, r0 + width // 2], [c0 + width // 2, c0, c0 + width // 2, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    if width == 5:
        rr = np.asarray([r0 + 4] * 5 + [r0 + 3] * 3 + [r0 + 2] * 3 + [r0 + 1] + [r0], dtype=np.int32)
        cc = np.asarray(list(range(c0, c0 + 5)) + list(range(c0 + 1, c0 + 4)) * 2 + [c0 + 2] * 2, dtype=np.int32)
        return rr, cc

    rr, cc = [r0, r0 + width - 1, r0 + width - 1], [c0 + width // 2, c0, c0 + width - 1]
    return skimage.draw.polygon(rr, cc, im_size)


def circle(r0, c0, width, im_size):
    if width == 5:
        rr = np.asarray([r0] + [r0 + 1] * 3 + [r0 + 2] * 5 + [r0 + 3] * 3 + [r0 + 4], dtype=np.int32)
        cc = np.asarray(
            [c0 + 2] + list(range(c0 + 1, c0 + 4)) + list(range(c0, c0 + 5)) + list(range(c0 + 1, c0 + 4)) + [c0 + 2],
            dtype=np.int32)
        return rr, cc

    radius = width // 2
    return skimage.draw.ellipse(
        r0 + radius, c0 + radius, radius, radius)


def cross(r0, c0, width, im_size):
    diff1 = width // 3 + 1
    diff2 = 2 * width // 3
    rr = [r0 + diff1, r0 + diff2, r0 + diff2, r0 + width, r0 + width,
          r0 + diff2, r0 + diff2, r0 + diff1, r0 + diff1, r0, r0, r0 + diff1]
    cc = [c0, c0, c0 + diff1, c0 + diff1, c0 + diff2, c0 + diff2, c0 + width,
          c0 + width, c0 + diff2, c0 + diff2, c0 + diff1, c0 + diff1]
    return skimage.draw.polygon(rr, cc, im_size)


def pentagon(r0, c0, width, im_size):
    diff1 = width // 3 - 1
    diff2 = 2 * width // 3 + 1
    rr = [r0 + width // 2, r0 + width, r0 + width, r0 + width // 2, r0]
    cc = [c0, c0 + diff1, c0 + diff2, c0 + width, c0 + width // 2]
    return skimage.draw.polygon(rr, cc, im_size)


def parallelogram(r0, c0, width, im_size):
    if width == 5:
        rr = np.asarray([r0] * 2 + [r0 + 1] * 3 + [r0 + 2] * 3 + [r0 + 3] * 3 + [r0 + 4] * 2, dtype=np.int32)
        cc = np.asarray(
            [c0, c0 + 1] + list(range(c0, c0 + 3)) + list(range(c0 + 1, c0 + 4)) + list(range(c0 + 2, c0 + 5)) + list(
                range(c0 + 3, c0 + 5)), dtype=np.int32)
        return rr, cc

    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0 + width // 2, c0 + width, c0 + width - width // 2]
    return skimage.draw.polygon(rr, cc, im_size)


def scalene_triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width // 2], [c0 + width - width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


class Push(gym.Env):
    GOAL = 'goal'
    STATIC_BOX = 'static_box'
    BOX = 'box'
    MOVING_BOXES_KEY = 'moving_boxes'

    STEP_REWARD = -0.01
    OUT_OF_FIELD_REWARD = -0.1
    COLLISION_REWARD = -0.1
    DEATH_REWARD = -1
    HIT_GOAL_REWARD = 1
    DESTROY_GOAL_REWARD = -1

    def __init__(self, n_boxes=5, n_static_boxes=0, n_goals=1, static_goals=True, width=5,
                 embodied_agent=False, return_state=True, observation_type='shapes', max_episode_steps=75,
                 border_walls=True, channels_first=True, channel_wise=False, channels_for_static_objects=True,
                 seed=None, render_scale=10, ternary_interactions=False,
                 ):
        if n_static_boxes > 0:
            assert n_goals == 0 or static_goals, 'Cannot have movable goals with static objects.'

        if n_goals > 0 and not static_goals:
            assert n_static_boxes == 0, 'Cannot have static objects with movable goals'

        self.w = width
        self.step_limit = max_episode_steps
        self.n_boxes = n_boxes
        self.embodied_agent = embodied_agent
        self.ternary_interactions = ternary_interactions

        self.goal_ids = set()
        self.static_box_ids = set()
        for k in range(self.n_boxes):
            box_id = self.n_boxes - k - 1
            if k < n_goals:
                self.goal_ids.add(box_id)
            elif k < n_goals + n_static_boxes:
                self.static_box_ids.add(box_id)
            else:
                break

        assert len(self.goal_ids) == n_goals
        assert len(self.static_box_ids) == n_static_boxes
        if self.embodied_agent:
            assert self.n_boxes > len(self.goal_ids) + len(self.static_box_ids)

        self.n_boxes_in_game = None
        self.static_goals = static_goals
        self.render_scale = render_scale
        self.border_walls = border_walls
        self.colors = get_colors(num_colors=max(9, self.n_boxes))
        self.observation_type = observation_type
        self.return_state = return_state
        self.channels_first = channels_first
        self.channel_wise = channel_wise
        self.channels_for_static_objects = channels_for_static_objects

        self.directions = {
            0: np.asarray((1, 0)),
            1: np.asarray((0, -1)),
            2: np.asarray((-1, 0)),
            3: np.asarray((0, 1))
        }
        self.direction2action = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}

        self.np_random = None

        if self.embodied_agent:
            self.action_space = spaces.Discrete(4)
        else:
            n_movable_objects = self.n_boxes - len(self.goal_ids) * self.static_goals - len(self.static_box_ids)
            self.action_space = spaces.Discrete(4 * n_movable_objects)

        if self.observation_type in ('squares', 'shapes'):
            observation_shape = (self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels())
            if self.channels_first:
                observation_shape = (observation_shape[2], *observation_shape[:2])
            self.observation_space = spaces.Box(0, 255, observation_shape, dtype=np.uint8)
        else:
            raise ValueError(f'Invalid observation_type: {self.observation_type}.')

        self.state = None
        self.steps_taken = 0
        self.pos = None
        self.image = None
        self.box_pos = np.zeros(shape=(self.n_boxes, 2), dtype=np.int32)

        self.seed(seed)
        self.reset()

    def _get_image_channels(self):
        if not self.channel_wise:
            return 3

        n_channels = self.n_boxes
        if not self.channels_for_static_objects:
            n_channels -= len(self.static_box_ids) + len(self.goal_ids) * self.static_goals

        return n_channels

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        if self.observation_type == 'squares':
            image = self.render_squares()
        elif self.observation_type == 'shapes':
            image = self.render_shapes()
        else:
            assert False, f'Invalid observation type: {self.observation_type}.'

        if self.return_state:
            return np.array(self.state), image

        return skimage.transform.resize(image, (3, 64, 64), order=0, anti_aliasing=False)

    def reset(self, *, seed=None, options=None):
        state = np.full(shape=[self.w, self.w], fill_value=-1, dtype=np.int32)

        # sample random locations for objects
        if self.embodied_agent:
            is_agent_in_main_area = self.np_random.random() > 4 * (self.w - 1) / self.w / self.w
            locs = self.np_random.choice((self.w - 2) ** 2, self.n_boxes - 1 + is_agent_in_main_area, replace=False)
            xs, ys = np.unravel_index(locs, [self.w - 2, self.w - 2])
            xs += 1
            ys += 1
            if not is_agent_in_main_area:
                agent_loc = self.np_random.choice(4 * (self.w - 1))
                side_id = agent_loc // (self.w - 1)
                if side_id == 0:
                    x = 0
                    y = agent_loc % (self.w - 1)
                elif side_id == 1:
                    x = agent_loc % (self.w - 1)
                    y = self.w - 1
                elif side_id == 2:
                    x = self.w - 1
                    y = self.w - 1 - agent_loc % (self.w - 1)
                elif side_id == 3:
                    x = self.w - 1 - agent_loc % (self.w - 1)
                    y = 0
                else:
                    raise ValueError(f'Unexpected side_id={side_id}')

                xs = np.append(x, xs)
                ys = np.append(y, ys)
        else:
            locs = self.np_random.choice(self.w ** 2, self.n_boxes, replace=False)
            xs, ys = np.unravel_index(locs, [self.w, self.w])

        # populate state with locations
        for i, (x, y) in enumerate(zip(xs, ys)):
            state[x, y] = i
            self.box_pos[i, :] = x, y

        self.state = state
        self.steps_taken = 0
        self.n_boxes_in_game = self.n_boxes - len(self.goal_ids) - int(self.embodied_agent)
        if not self.embodied_agent:
            self.n_boxes_in_game -= len(self.static_box_ids) * int(self.static_goals)

        return self._get_observation()

    def _get_type(self, box_id):
        if box_id in self.goal_ids:
            return Push.GOAL
        elif box_id in self.static_box_ids:
            return Push.STATIC_BOX
        else:
            return Push.BOX

    def _destroy_box(self, box_id):
        box_pos = self.box_pos[box_id]
        self.state[box_pos[0], box_pos[1]] = -1
        self.box_pos[box_id] = -1, -1
        if self._get_type(box_id) == Push.BOX or (
                not self.embodied_agent and self._get_type(box_id) == Push.STATIC_BOX):
            self.n_boxes_in_game -= 1

    def _move(self, box_id, new_pos):
        old_pos = self.box_pos[box_id]
        self.state[old_pos[0], old_pos[1]] = -1
        self.state[new_pos[0], new_pos[1]] = box_id
        self.box_pos[box_id] = new_pos

    def _is_free_cell(self, pos):
        return self.state[pos[0], pos[1]] == -1

    def _get_occupied_box_id(self, pos):
        return self.state[pos[0], pos[1]]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        vec = self.directions[action % len(self.directions)]
        box_id = action // len(self.directions)
        box_old_pos = self.box_pos[box_id]
        box_new_pos = box_old_pos + vec
        box_type = self._get_type(box_id)

        done = False
        reward = Push.STEP_REWARD
        moving_boxes = [0] * self.n_boxes
        moving_boxes[box_id] = 1

        if not self._is_in_grid(box_old_pos, box_id):
            # This box is out of the game. There is nothing to do.
            pass
        elif not self._is_in_grid(box_new_pos, box_id):
            reward += Push.OUT_OF_FIELD_REWARD
            if not self.border_walls:
                # push out of grid, destroy object or finish episode if an agent is out of the grid
                if box_type == Push.GOAL:
                    reward += Push.DESTROY_GOAL_REWARD
                elif self.embodied_agent:
                    reward += Push.DEATH_REWARD
                    done = True

                self._destroy_box(box_id)
        elif not self._is_free_cell(box_new_pos):
            # push into another box
            another_box_id = self._get_occupied_box_id(box_new_pos)
            another_box_type = self._get_type(another_box_id)

            if box_type == Push.BOX:
                if another_box_type == Push.BOX:
                    if self.ternary_interactions:
                        moving_boxes[another_box_id] = 1
                        another_box_new_pos = box_new_pos + vec
                        if self._is_in_grid(another_box_new_pos, another_box_id):
                            if self._is_free_cell(another_box_new_pos):
                                self._move(another_box_id, another_box_new_pos)
                                self._move(box_id, box_new_pos)
                            elif self._get_type(self._get_occupied_box_id(another_box_new_pos)) == Push.GOAL:
                                reward += Push.HIT_GOAL_REWARD
                                self._destroy_box(another_box_id)
                                self._move(box_id, box_new_pos)
                            else:
                                reward += Push.COLLISION_REWARD
                        else:
                            reward += Push.OUT_OF_FIELD_REWARD
                            if not self.border_walls:
                                self._destroy_box(another_box_id)
                                self._move(box_id, box_new_pos)
                    else:
                        reward += Push.COLLISION_REWARD
                elif another_box_type == Push.GOAL:
                    if self.embodied_agent:
                        reward += Push.COLLISION_REWARD
                    else:
                        reward += Push.HIT_GOAL_REWARD
                        self._destroy_box(box_id)
                else:
                    assert another_box_type == Push.STATIC_BOX
                    reward += Push.COLLISION_REWARD
            elif box_type == Push.GOAL:
                if another_box_type in (Push.BOX, Push.STATIC_BOX):
                    self._destroy_box(another_box_id)
                    self._move(box_id, box_new_pos)
                    reward += Push.HIT_GOAL_REWARD
                else:
                    assert another_box_type == Push.GOAL
                    reward += Push.COLLISION_REWARD
            else:
                assert False, f'Cannot move a box of type:{box_type}'
        else:
            # pushed into open space, move box
            self._move(box_id, box_new_pos)

        self.steps_taken += 1
        if self.steps_taken >= self.step_limit:
            done = True

        if self.n_boxes_in_game == 0:
            done = True

        return self._get_observation(), reward, done, {Push.MOVING_BOXES_KEY: moving_boxes}

    def _is_in_grid(self, point, box_id):
        if not self.embodied_agent or box_id == 0:
            return (0 <= point[0] < self.w) and (0 <= point[1] < self.w)

        return (1 <= point[0] < self.w - 1) and (1 <= point[1] < self.w - 1)

    def print(self, message=''):
        state = self.state
        chars = {-1: '.'}
        for box_id in range(self.n_boxes):
            if box_id in self.goal_ids:
                chars[box_id] = 'x'
            elif box_id in self.static_box_ids:
                chars[box_id] = '#'
            else:
                chars[box_id] = '@'

        pretty = "\n".join(["".join([chars[x] for x in row]) for row in state])
        print(pretty)
        print("TIMELEFT ", self.step_limit - self.steps_taken, message)

    def clone_full_state(self):
        sd = copy.deepcopy(self.__dict__)
        return sd

    def restore_full_state(self, state_dict):
        self.__dict__.update(state_dict)

    def get_action_meanings(self):
        return ["down", "left", "up", "right"] * (
                self.n_boxes - len(self.static_box_ids) - len(self.goal_ids) * self.static_goals)

    def render_squares(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels()),
                      dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            rr, cc = square(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            if self.channel_wise:
                if not self.channels_for_static_objects and (idx in self.goal_ids or idx in self.static_box_ids):
                    im[rr, cc] = 1
                else:
                    im[rr, cc, idx] = 1
            else:
                im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255

        return im.astype(dtype=np.uint8)

    def render_shapes(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels()),
                      dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            shape_id = idx % 8
            if shape_id == 0:
                rr, cc = circle(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 1:
                rr, cc = triangle(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 2:
                rr, cc = square(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 3:
                rr, cc = parallelogram(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 4:
                rr, cc = cross(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 5:
                rr, cc = diamond(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 6:
                rr, cc = pentagon(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            else:
                rr, cc = scalene_triangle(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)

            if self.channel_wise:
                if not self.channels_for_static_objects and (idx in self.goal_ids or idx in self.static_box_ids):
                    im[rr, cc] = 1
                else:
                    im[rr, cc, idx] = 1
            else:
                im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255
        return im.astype(np.uint8)
