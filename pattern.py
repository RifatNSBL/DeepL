import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tileSize):
        self.resolution = resolution
        self.tileSize = tileSize
        self.output = None

    def draw(self):
        tilePerAxis = self.resolution // self.tileSize
        grid = np.zeros((tilePerAxis, tilePerAxis), dtype=int)
        grid[::2, ::2] = 1
        grid[1::2, 1::2] = 1
        self.output = grid
        return np.copy(grid)

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
        x_coords = np.arange(self.resolution)
        y_coords = np.arange(self.resolution)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        distance = np.sqrt((x_grid - self.position[0])**2 + (y_grid - self.position[1])**2)
        image = np.where(distance <= self.radius, 255, 0).astype(np.uint8)
        
        return image


    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


if __name__ == "__main__":
    checker = Checker(10,2)
    circle = Circle(200,50,(100, 50))
    plot = checker.show()
    plot2 = circle.show()