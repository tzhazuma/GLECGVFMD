import numpy as np
import scipy as sp
import pandas as pd
import cv2 as cv
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import sklearn as sk
import torchvision as tv
import torchvision.transforms as transforms
import skimage as ski
from torch.utils.data import DataLoader,Dataset
import os.path as path
import os
import PIL.Image as Image
def getdevice():
    device="cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device="mps"
    elif torch.xpu.is_available():
        device="xpu"
    return device