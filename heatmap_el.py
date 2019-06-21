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

#from . import log; log = log.getChild(__name__)
def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi

#input_filename="/eos/user/m/mociduki/tau_sample/v14/images_new_1p0n.h5"
input_filename="/eos/user/l/lehrke/Data/Data/2019-05-02/MC_abseta_0.0_1.3_et_0.0_1000000.0_processes_pid.h5"
print "input filename",input_filename

f = h5py.File(input_filename, 'r')

#retrieve dataset
dset = f['train']

# sys.exit()
# print "it shouldn't show this."

#dump variable names

#dump the matrix

suffix="electron"
fixed_scale=True
irec=int(sys.argv[1]) #event number index
cal_layer=int(sys.argv[2])

print dset.keys()

layer_name='em_barrel_Lr%d'%cal_layer
if cal_layer>3: 
    cal_layer=cal_layer-3
    layer_name='tile_barrel_Lr%d'%cal_layer
    pass

if cal_layer==6: layer_name='tracks'

image_3d= dset[layer_name][irec]
print image_3d

tru_eta= dset["p_truth_eta"][irec]
rec_eta= dset["p_eta"][irec]

tru_phi= dset["p_truth_phi"][irec]
rec_phi= dset["p_phi"][irec]

tru_pt = dset["p_truth_pt"][irec]
rec_pt = dset["p_et_calo"][irec]/1e3 # MeV to GeV

rec_e  = dset["p_e"][irec]/1e3 # MeV to GeV

m=0
n=0

# if   cal_layer==1: m=4; n=128
# elif cal_layer==2: m=32;n=32
# elif cal_layer==3: m=32;n=16
# elif cal_layer==4: m=16;n=16
# elif cal_layer==5: m=16;n=16
# elif cal_layer==6: m=15;n=4
# if   cal_layer==1: m=11;n=56
# elif cal_layer==2: m=11;n=56
# elif cal_layer==3: m=11;n=56
# elif cal_layer==4: m=11;n=56
# elif cal_layer==5: m=11;n=56
# elif cal_layer==6: m=11;n=56

image_2d=image_3d.transpose()/rec_e

print image_2d.shape, type(image_2d)

# harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
# print type(harvest)

fig = plt.figure()
# if fixed_scale:
#     image[image <= 0] = 0.00001

#plt.imshow(image_2d)
plt.imshow(
    image_2d, 
    extent=[-0.0875, 0.0875, -0.13499031, 0.13499031], 
    interpolation='nearest',
    cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
    norm=None if fixed_scale is False else LogNorm(0.0001, 1))

plt.colorbar()
# plt.plot(
#     rec_eta - tru_eta, dphi(rec_phi, tru_phi), 'ro', 
#     label='True pT = %1.2f GeV' % (tru_pt / 1000.) )

# plt.plot(
#     rec['off_tracks_eta'][0] - pos_central_cell['eta'], 
#     dphi(rec['off_tracks_phi'][0], pos_central_cell['phi']), 'bo', 
#     label='reco charge pi, pT = %1.2f GeV' % (rec['off_tracks_pt'][0] / 1000.))

# if not '0n' in suffix:
#     plt.plot(
#         rec['true_neutral_eta'] - rec['true_eta'], 
#         dphi(rec['true_neutral_phi'], rec['true_phi']), 'g^', 
#         label='true neutral pi, pT = %1.2f GeV' % (rec['true_neutral_pt'] / 1000.))
#     pass
plt.xlabel('eta')
plt.ylabel('phi')
plt.title('{0}: image {1} sampling {2}'.format(suffix, irec, cal_layer)+ ", Rec pT = %1.2f GeV"% (rec_pt))
plt.legend(loc='upper right', fontsize='small', numpoints=1)

outputfile= 'plots/imaging/images/image_el{0}_{1}_{2}.png'.format(irec, layer_name, suffix)
print outputfile
fig.savefig(outputfile)
fig.clf()
plt.close()

#dump values of all variables
#list(dset["table_0"])
