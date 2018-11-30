from scipy.spatial import ConvexHull
import numpy as np

class ConvexHullPolicy():
    def __init__(self, img_size):
        self.done = True
        self.pen_up = False
        self.img_size = img_size
        self.mask = None

    def get_action(self, state, true_segmentation):
        if self.mask is None or not np.array_equal(true_segmentation, self.mask):
            self.done = True
            self.pen_up = False

        if self.pen_up:
            self.pen_up = False
            self.done = True
            return 1 # Finish drawing

        if self.done:
            self.mask = true_segmentation
            assert(self.mask.shape == state.shape[:2])
            points = np.argwhere(true_segmentation)
            if len(points) == 0:
                return 1
            hull = ConvexHull(points)
            self.vertices = [points[vertex] for vertex in hull.vertices]
            self.i = 0
            self.done = False        

        if self.i >= len(self.vertices):
            self.pen_up = True
            return 0 # Pen Up

        x, y = self.vertices[self.i]
        action = int(2 + x * self.img_size + y)

        self.i += 1
        return action