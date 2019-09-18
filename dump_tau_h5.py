import h5py
import numpy as np

import math
import numpy as np
import itertools
import socket
import sys

import matplotlib as mpl;

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LogNorm

input_filename=sys.argv[1]
print "input filename=",input_filename

f = h5py.File(input_filename, 'r')

#retrieve dataset
dset = f['data/table_0']

for attr in dset.attrs: 
    try:
        if dset.attrs[attr]==0: break
        #print dset.attrs[attr]
        print dset.attrs[attr], "=", dset[dset.attrs[attr]]
    except:
        print "EXCEPTION: no right value for", attr
        
#print dset.attrs['FIELD_0_NAME']
    
sys.exit(0)
