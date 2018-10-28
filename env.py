import numpy as np
from scipy.ndimage.filters import gaussian_filter

PEN_DOWN = 0
PEN_UP = 1
FINISH = 2

class Environment():
    # Pulls in new images with generator_fn
    # generator_fn should return a preprocessed image and a segmentation mask
    def __init__(self, generator_fn, gaussian_std=2.0, img_shape=(256,256)):
        self.generator_fn = generator_fn
        self.gaussian_std = gaussian_std
        self.img_shape = img_shape

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
        pass
    
    # Returns initial state
    def reset(self):
        self.curr_image, self.curr_mask = self.generator_fn()
        assert(self.curr_image.shape == self.img_shape)
        assert(self.curr_mask.shape == self.img_shape)

        self.curr_blurred_mask = gaussian_filter(self.curr_mask, self.gaussian_std)
        self.state_map = np.zeros((3, self.img_shape[0], self.img_shape[1]))

        self.last_action = PEN_UP


        