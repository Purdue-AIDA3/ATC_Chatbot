# This file collects all the relevant libraries that we need
# for the GPT model
# This file can be run as a standalone script.

import matplotlib.pyplot as plt
import os
import requests

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import argparse
import json
import numpy as np


import requests
import tensorflow as tf
from tqdm import tqdm


import zipfile
from pathlib import Path
import re
import time

import pandas as pd
from functools import partial
from importlib.metadata import version

import urllib.request
