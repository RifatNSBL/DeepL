import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tileSize):
        self.resolution = resolution
        self.tileSize = tileSize
        self.output = None

    def draw(self):
        ''' 
        Let's see an example:
        5,0     5,1     5,2  !  5,3     5,4     5,5  ! 
        4,0     4,1     4,2  !  4,3     4,4     4,5  ! 
        3,0     3,1     3,2  !  3,3     3,4     3,5  ! 
        ----------------------------------------------
        2,0     2,1     2,2  !   2,3     2,4     2,5 ! 
        1,0     1,1     1,2  !   1,3     1,4     1,5 ! 
        0,0     0,1     0,2  !   0,3     0,4     0,5 ! 
        '''

        x_coords = np.arange(self.resolution) # resolution of image w.r.t. axis
        y_coords = np.arange(self.resolution)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords) # define a grid, basically matrices of indices for x and y axes
        x_cond = np.where((x_grid // self.tileSize) % 2 == 0, 1, 0) ## check if int part of index x / tilesize is even
        y_cond = np.where((y_grid // self.tileSize) % 2 == 0, 1, 0) ## check if int part of index y / tilesize is even
        condition = (x_cond) ^ (y_cond) # xor condition -> returns 1 matrix of dim(res, res) where cells with value 1
        #correspond to cells that should be painted
        grid = np.where(condition, 1, 0)
        self.output = grid
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
    
    def draw(self):
        x_coords = np.arange(self.resolution) # resolution of image w.r.t. axis
        y_coords = np.arange(self.resolution)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords) # define a grid
        # for each pixel find L2-distance from the center of circle
        distance = np.sqrt((x_grid - self.position[0])**2 + (y_grid - self.position[1])**2) 
        # if pixel inside the circle - paint it black
        '''
        # image = np.where(distance <= self.radius, 255, 0).astype(np.uint8) 
        NOTE: this method doesnt pass the test because based on test hints 
        The desired output is a boolean array and not a binary array.
        '''
        image = distance <= self.radius
        
        return image


    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None
    
    def draw(self):
        # Slow solution, manually create matrices for each pixel
        
        # red_mesh = np.tile(red_channel, self.resolution)
        # blue_mesh = np.tile(blue_channel, self.resolution)
        # green_mesh = np.tile(green_channel, self.resolution)

        # red_mesh = np.reshape(red_mesh, (self.resolution, self.resolution))
        # blue_mesh = np.reshape(blue_mesh, (self.resolution, self.resolution))
        # green_mesh = np.reshape(green_mesh, (self.resolution, self.resolution)).T

        # spectrum[:, :, 0] = red_mesh
        # spectrum[:, :, 1] = green_mesh
        # spectrum[:, :, 2] = blue_mesh
        
        spectrum = np.zeros([self.resolution, self.resolution, 3])

        red_channel = np.linspace(0, 1, self.resolution) # resolution of image w.r.t. axis
        blue_channel = np.linspace(1, 0, self.resolution)
        green_channel = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)

        # Fast solution: let numpy repeat given vector in given dimension: R - 0, G - 1, B - 2
        spectrum[:, :, 0] = red_channel
        spectrum[:, :, 1] = green_channel
        spectrum[:, :, 2] = blue_channel

        self.output = spectrum
        return np.copy(self.output)


    def show(self):
        plt.imshow(self.draw())
        plt.show()

        