import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage import feature

PEN_DOWN = 2
PEN_UP = 0
FINISH = 1

class Environment():
    # Pulls in new images with generator_fn
    # generator_fn should return a preprocessed image and a segmentation mask
    def __init__(self, generator, gaussian_std=2.0, img_shape=(256,256), alpha=0.05, max_line_len=50):
        self.generator = generator
        self.gaussian_std = gaussian_std
        self.img_shape = img_shape
        self.alpha = alpha
        self.max_line_len = max_line_len

        self.curr_image = None
        self.curr_mask = None
        self.curr_blurred_mask = None
        self.state_map = None
        self.last_action = None
        self.first_vertex = None

        self.reset()

    # Returns (new_state, reward, done)
    # Action should be int: 0 = pen up, 1 = finish, other = index into array
    def step(self, action):
        action_class = PEN_DOWN if action > 1 else action
        coord_x, coord_y = (-1,-1)
        if action_class == PEN_DOWN:
            coord_x = action // self.img_shape[0]
            coord_y = action % self.img_shape[0]

        if self.last_action == PEN_UP:
            if action_class == PEN_UP:
                return self._get_state(), -1.0, False
            elif action_class == PEN_DOWN:
                self.first_vertex = (coord_x, coord_y)
                self.state_map[2,:,:] = 0
                self.state_map[2, coord_x, coord_y] = 1
                self.state_map[1, coord_x, coord_y] = 1
                rew = self.curr_blurred_mask[coord_x, coord_y] / self.alpha
                return self._get_state(), rew, False
            else:
                return self._get_state(), -1.0, True
        elif self.last_action == PEN_DOWN:
            if action_class == PEN_UP:
                rew = self._finish_polygon(coord_x, coord_y)
                self.first_vertex = None
                return self._get_state(), rew, False 
            
            elif action_class == PEN_DOWN:
                prev_vertex_x, prev_vertex_y = np.where(self.state_map[3] == 1)
                prev_vertex_x = prev_vertex_x[0]
                prev_vertex_y = prev_vertex_y[0]

                # Penalize illegal placements
                #if np.hypot(coord_x - prev_vertex_x, coord_y - prev_vertex_y) > self.max_line_len:
                #    return self._get_state(), -1, False

                line_x, line_y = self._get_line_coordinates(prev_vertex_x, prev_vertex_y, coord_x, coord_y)
                rew = self._contour_reward(line_x, line_y)
                for x, y in zip(line_x, line_y):
                    self.state_map[1, x, y] = 1
                
                self.state_map[2, prev_vertex_x, prev_vertex_y] = 0
                self.state_map[2, coord_x, coord_y] = 1

                return self._get_state(), rew, False

            else:
                rew = self._finish_polygon(coord_x, coord_y)
                return self._get_state(), rew, True

        else:
            raise Exception('Environment is done, should have been reset') 
    
    # Returns initial state
    def reset(self):
        self.curr_image, self.curr_mask = next(self.generator)
        if len(self.curr_mask.shape) == 3:
            self.curr_mask = self.curr_mask[:,:,0]
        assert(self.curr_image.shape == self.img_shape)
        assert(self.curr_mask.shape == self.img_shape[:2])

        mask_outline = feature.canny(self.curr_mask.astype(np.float32), sigma=2)
        self.curr_blurred_mask = gaussian_filter(mask_outline, self.gaussian_std)
        self.curr_mask = self.curr_mask.astype(np.bool_)
        self.state_map = np.zeros((3, self.img_shape[0], self.img_shape[1]), dtype=np.int16)

        self.last_action = PEN_UP
        self.first_vertex = None

        first_state = self._get_state()
        return first_state

    def _get_state(self):
        return np.concatenate((self.curr_image, np.transpose(self.state_map)), axis=-1)
    
    def _contour_reward(self, line_x, line_y):
        rew = 0.0
        for x, y in zip(line_x, line_y):
            rew += self.curr_blurred_mask[x, y]
            self.curr_blurred_mask[x, y] = 0.0 # Can't get contour reward twice
        return rew / self.alpha
    
    def _region_reward(self):
        assert(self.curr_mask.dtype == np.bool_)
        mask = self.state_map[1].astype(np.bool_)
        intersection = (mask * self.curr_mask).sum()
        union = (mask + self.curr_mask).sum()
        iou = float(intersection) / float(union)
        return iou

    def _get_line_coordinates(self, x0, y0, x1, y1):
        length = int(np.hypot(x1 - x0, y1 - y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        return x.astype(np.int), y.astype(np.int)
    
    # Returns contour reward + region reward for a finished polygon
    def _finish_polygon(self, last_x, last_y):
        last_line_x, last_line_y = self._get_line_coordinates(last_x, last_y, self.first_vertex[0], self.first_vertex[1])
        rew = self._contour_reward(last_line_x, last_line_y)
        for x, y in zip(last_line_x, last_line_y):
            self.state_map[1, x, y] = 1
        # Fill in polygon
        self.state_map[1] = binary_fill_holes(self.state_map[1])
        rew += self._region_reward()

        # Add polygon to overall segmentation mask
        polys = self.state_map[0]
        polys += self.state_map[1]
        polys[polys > 1] = 1

        self.state_map[1,:,:] = 0
        self.state_map[2,:,:] = 0

        return rew
