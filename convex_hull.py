from scipy.spatial import ConvexHull

class ConvexHullPolicy():
    def __init__(self, img_size):
        self.done = True
        self.pen_up = False
        self.img_size = img_size
        self.mask = None

    def get_action(self, state, true_segmentation):
        if true_segmentation != self.mask:
            self.done = True
            self.pen_up = False

        if self.pen_up:
            self.pen_up = False
            return 1 # Finish drawing

        if self.done:
            self.mask = true_segmentation
            assert(self.mask.shape == state.shape)
            self.hull = ConvexHull(true_segmentation)
            self.i = 0
            self.done = False        

        if self.i >= len(self.hull.points):
            self.done = True
            return 0 # Pen Up

        x, y = self.hull.points[self.i]
        action = 2 + x * self.img_size + y

        self.i += 1
        return action