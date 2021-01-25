from __future__ import print_function
from __future__ import division
from rpi_ws281x import *

import platform
import numpy as np
import config

# Raspberry Pi controls the LED strip directly
strip = Adafruit_NeoPixel(config.N_PIXELS, config.LED_PIN,
                          config.LED_FREQ_HZ, config.LED_DMA,
                          config.LED_INVERT, config.BRIGHTNESS)

pixels = np.tile(1, (3, config.N_PIXELS))
"""Pixel values for the LED strip"""

_is_python_2 = int(platform.python_version_tuple()[0]) == 2
