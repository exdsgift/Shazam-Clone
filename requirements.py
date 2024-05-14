import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy import fft, signal
from scipy.io.wavfile import read

from scipy.fft import fftfreq

# Database
import glob
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle