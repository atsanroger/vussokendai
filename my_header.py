##
##        @file    my_header
##        @brief
##        @author  Wei-Lun Chen (wlchen)
##                 $LastChangedBy: wlchen $
##        @date    $LastChangedDate: 2024-10-17 16:42:21 #$
##        @version $LastChangedRevision: 2499 $
##

# --- Basic Mathematics or Scientific Computation ---
import math  # Basic math operations
import cmath  # Complex number math operations
import numpy as np  # Numerical operations with arrays, matrices, etc.
import cupy as cp # Numerical operations with GPU
from scipy import special as spsp  # Special functions like gamma, beta, etc.
from scipy.integrate import quad

# --- Uncertainty and Statistical Analysis ---
import gvar as gv  # Handling data with uncertainties
import lsqfit  # Least-squares fitting with uncertainties

# --- Plotting and Visualization ---
import matplotlib.pyplot as plt  # Plotting library for visualizing data
from matplotlib.pyplot import figure  # Figure creation for plots

# --- File I/O and Data Serialization ---
import argparse
import pickle  # Python object serialization
import h5py  # Handling HDF5 file format for large data storage
import pandas as pd
import time

# --- Miscellaneous Utilities ---
import warnings  # Warning control
import subprocess  # Running external processes and shell commands
import sys  # System-specific parameters and functions
import os

# --- Multithreading and Parallelism ---
import threading  # Thread-based parallelism
from concurrent.futures import ThreadPoolExecutor  # High-level interface for multithreading
import multiprocessing as mp  # Process-based parallelism
num_cores = mp.cpu_count()  # Get the number of CPU cores


# --- Define Particle Constants ---

# Tau lepton mass with uncertainty (タウレプトン質量と不確定性)
mtau = gv.gvar(1.77686, 0.00018)

# Constant Rcont, proportional to tau mass (Rcont 定数、タウ質量に比例)
Rcont = 12 * np.pi ** 2 / mtau ** 2

# CKM matrix elements (CKM行列要素)
VUSPDGIN = gv.gvar(0.2216, 0.0015)  # PDG input
VUSPDGEX = gv.gvar(0.2243, 0.0005)  # PDG experimental value

# Electroweak correction factor (電弱補正係数)
SEW = gv.gvar(1.0201, 0.00001)

# Fermi constant (フェルミ定数)
GF = gv.gvar(1.1663787e-05, 6e-10)

# Decay width for tau to electron (タウから電子への崩壊幅)
DecayBasic = mtau ** 5 * GF ** 2 / (192 * np.pi ** 3)

# Total tau decay width (タウの総崩壊幅)
TotalWidth = gv.gvar(2.266e-12, 0)

# Decay width excluding non-kaon channels (非カオンチャネルを除外した崩壊幅)
totaldecaywidth = DecayBasic / gv.gvar(0.1779, 0.00009)

# Normalized total decay width (正規化された総崩壊幅)
Total = 2 * 10 ** (-12) / gv.gvar(0.01779, 0.00009)

# Prefactor connecting Gapp to decay width (崩壊幅に接続する前置係数)
Prefactor = GF ** 2 * mtau ** 3 / (16 * np.pi)

# Decay probabilities for specific channels (特定チャネルの崩壊確率)
PDGKaon  = 0.00708743  # Kaon channel
PDGKstar = 0.0140239   # K* channel
PDGPion  = 0.113876    # Pion channel

# Lattice Parameters
Inverse_a48   = gv.gvar(1.7295,0.0038)
Inverse_a64   = gv.gvar(2.3586,0.0007)
Mass_kaon_48  = gv.gvar(0.28853,0.00014)
Mass_kaon_64  = gv.gvar(0.21531,0.00017)

Leng_48= gv.gvar(5.468,0.012) #unit fm
Leng_64= gv.gvar(5.349,0.016) #unit fm

ZA_48=0.71076
ZA_64=0.74293

fk_48=gv.gvar(0.090396,0.000086)
fk_64=gv.gvar(0.066534,0.000099)
