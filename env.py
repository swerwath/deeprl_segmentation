import numpy as np
from scipy.ndimage.filters import gaussian_filter

PEN_DOWN = 0
PEN_UP = 1
FINISH = 2

class Environment():
    # Pulls in new images with generator_fn
    # generator_fn should return a preprocessed image and a segmentation mask
    def __init__(self, generator_fn, gaussian_std=2.0, img_shape=(256,256), alpha=0.05):
        self.generator_fn = generator_fn
        self.gaussian_std = gaussian_std
        self.img_shape = img_shape
        self.alpha = 0.05

        self.curr_image = None
        self.curr_mask = None
        self.curr_blurred_mask = None
        self.state_map = None
        self.last_action = None

        self.reset()

    # Returns (new_state, reward, done)
    # Action should be (int, (int, int))
    # First int is 0 = pen down, 1 = pen up, 2 = finish 
    # Int tuple is coordinates for pen down
    # New state is 
    def step(self, action):
        action_class, (coord_x, coord_y) = action
        if self.last_action == PEN_UP:
            if action_class == PEN_UP:
                return self._get_state(), -1.0, False
            elif action_class == PEN_DOWN:
                self.state_map[2,:,:] = 0
                self.state_map[2, coord_x, coord_y] = 1
                rew = self.curr_blurred_mask[coord_x, coord_y] / self.alpha
                return self._get_state(), rew, False
            else:
                return self._get_state(), -1.0, True
        elif self.last_action == PEN_DOWN:
            # TODO
            pass
        else:
            raise Exception('Environment is done, should have been reset') 
    
    # Returns initial state
    def reset(self):
        self.curr_image, self.curr_mask = self.generator_fn()
        assert(self.curr_image.shape == self.img_shape)
        assert(self.curr_mask.shape == self.img_shape)

        self.curr_blurred_mask = gaussian_filter(self.curr_mask, self.gaussian_std)
        self.state_map = np.zeros((3, self.img_shape[0], self.img_shape[1]), dtype=np.int16)

        self.last_action = PEN_UP

    def _get_state(self):
        return np.concatenate((self.curr_image, self.state_map))
    
    def _contour_reward(self, line_x, line_y):
        rew = 0.0
        for x, y in zip(line_x, line_y):
            rew += self.curr_blurred_mask[x, y]
        return rew / self.alpha

    def _get_line_coordinates(self, x0, y0, x1, y1):
        length = int(np.hypot(x1 - x0, y1 - y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        return x.astype(np.int), y.astype(np.int) 