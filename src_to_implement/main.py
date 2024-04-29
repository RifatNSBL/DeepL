import numpy as np
import matplotlib as plt
from pattern import Checker, Circle, Spectrum

if __name__ == "__main__":
    checker = Checker(20, 4)
    circle = Circle(512, 20, (50, 50))
    spectrum = Spectrum(123)
    # checker.show()
    circle.show()
    # spectrum.show()