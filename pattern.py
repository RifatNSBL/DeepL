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
        x_grid, y_grid = np.meshgrid(x_coords, y_coords) # define a grid#
        x_cond = np.where((x_grid // self.tileSize) % 2 == 0, 1, 0) ## check if int part of index x / tilesize is even
        y_cond = np.where((y_grid // self.tileSize) % 2 == 0, 1, 0) ## check if int part of index y / tilesize is even
        condition = (x_cond) ^ (y_cond) # xor condition -> returns 1 matrix of dim(res, res) where cells with value 1
        #correspond to cells that should be painted
        grid = np.where(condition, 0, 1)
        self.output = grid
        return np.copy(grid)

    def show(self):
        im = plt.imshow(self.draw(), cmap='gray', extent=[0, self.resolution, 0, self.resolution])
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
        x_grid, y_grid = np.meshgrid(x_coords, y_coords) # define a grid#
        distance = np.sqrt((x_grid - self.position[0])**2 + (y_grid - self.position[1])**2) 
        image = np.where(distance <= self.radius, 255, 0).astype(np.uint8)
        
        return image


    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


if __name__ == "__main__":
    checker = Checker(12, 3)
    #circle = Circle(200,50,(100, 50))
    plot = checker.show()
    #plot2 = circle.show()