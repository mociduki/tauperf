import h5py
import tables
import numpy as np

from ROOT import TLorentzVector

import math
import itertools
import socket
import sys

import matplotlib as mpl;

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LogNorm

import datetime

### 0,1,2= high pt signal, low pt signal, background
targetType=int(sys.argv[1])
if targetType>4:
    print "Currently target type is supported only up to 4, aborting!!"
    sys.exit()
    pass
#cal_layer=int(sys.argv[2])

nClasses=int(sys.argv[2])
    
#from . import log; log = log.getChild(__name__)
def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi

class Image(tables.IsDescription):
#     s1 = tables.Float64Col(shape=(4, 128), dflt=0.0)
#     s2 = tables.Float64Col(shape=(32, 32), dflt=0.0)
#     s3 = tables.Float64Col(shape=(32, 16), dflt=0.0)
#     s4 = tables.Float64Col(shape=(16, 16), dflt=0.0)
#     s5 = tables.Float64Col(shape=(16, 16), dflt=0.0)

    s0 = tables.Float32Col(shape=(11, 56), dflt=0.0)
    #N-th layer of the EM-calorimeter cells
    s1 = tables.Float32Col(shape=(11, 56), dflt=0.0)
    s2 = tables.Float32Col(shape=(11, 56), dflt=0.0)
    s3 = tables.Float32Col(shape=(11, 56), dflt=0.0)
    #N-th layer of the Tile-calorimeter cells
    s4 = tables.Float32Col(shape=(11,  7), dflt=0.0)
    s5 = tables.Float32Col(shape=(11,  7), dflt=0.0)
    s6 = tables.Float32Col(shape=(11,  7), dflt=0.0)
    tracks = tables.Float64Col(shape=(15, 4))

    # kinematic info of the electron
    e = tables.Float64Col()  
    pt = tables.Float64Col() 
    eta = tables.Float64Col()
    phi = tables.Float64Col()
    mu = tables.Float64Col() # pile up = average number of interactions per bunch crossing
    # output of the existing algorithm
    pantau = tables.Float64Col() 
    llh_tight = tables.Float64Col()
    llh_medium = tables.Float64Col()
    llh_loose = tables.Float64Col()
    truthmode = tables.Float64Col() # true label of the categories
    # kinematic information of true electron
    true_pt = tables.Float64Col()   
    true_eta = tables.Float64Col()
    true_phi = tables.Float64Col()
    true_m = tables.Float64Col()


    Rhad     = tables.Float64Col()
    f3       = tables.Float64Col()
    weta2    = tables.Float64Col()

    Rphi     = tables.Float64Col()
    Reta     = tables.Float64Col()
    Eratio   = tables.Float64Col()

    f1       = tables.Float64Col()
    d0       = tables.Float64Col()
    d0Sig    = tables.Float64Col()

    dPoverP  = tables.Float64Col()
    TRTPID   = tables.Float64Col()
    deltaEta1= tables.Float64Col()

    dPhiRes2 = tables.Float64Col()
    nSCT     = tables.Float64Col()

    alpha_e = tables.Float64Col() # not used for the moment
    pass

def tau_tracks_simple(rec, event_idx, el_4v):
    """
    Laser was here.
    """
    maxtracks = 15

    imp   = []
    deta  = []
    dphi  = []
    d0    = []

    trk_idx=0

    trks_p     = np.array([])
    trks_deta  = np.array([])
    trks_dphi  = np.array([])
    trks_d0    = np.array([])

    for trk_pt in rec['tracks_pt'][event_idx]:
        
        trk_phi = rec['tracks_phi'][event_idx][trk_idx]
        trk_eta = rec['tracks_eta'][event_idx][trk_idx]
        trk_d0  = rec['tracks_d0' ][event_idx][trk_idx]
        
        trk_4v = TLorentzVector()
        trk_4v.SetPtEtaPhiM(trk_pt,trk_eta,trk_phi,139.57061) # assume tracks are charged pions

        # print trk_idx,":",trk_pt, ", p=",        trk_4v.P()

        if abs(trk_4v.Eta()-el_4v.Eta())>=999: continue
        trks_p    = np.append(trks_p,     trk_4v.P())
        trks_deta = np.append(trks_deta,  abs(trk_4v.Eta()-el_4v.Eta()) )
        trks_dphi = np.append(trks_dphi,  el_4v.DeltaPhi(trk_4v) )
        trks_d0   = np.append(trks_d0,    trk_d0)

        trk_idx+=1
        if trk_idx==maxtracks: break
        pass

    # indices = np.where(rec['off_tracks_deta'] > -1000)
    indices = np.where(trks_deta > -1000)

    rp     = trks_p             .take(indices[0])
    rdeta  = trks_deta          .take(indices[0])
    rdphi  = trks_dphi          .take(indices[0])
    rd0    = trks_d0            .take(indices[0])

    # electron energy
    el_ene = el_4v.E()#rec['off_pt'][event_idx] * np.cosh(rec['off_eta'][event_idx] )

    # for (p, e, f, c) in zip(rp, rdeta, rdphi, rclass):
    for (p, e, f, d) in zip(rp, rdeta, rdphi, rd0):
        imp .append(p / el_ene)
        deta.append(e)
        dphi.append(f)
        d0  .append(d)
        pass

    imp   += [0] * (maxtracks - len(imp ))
    deta  += [0] * (maxtracks - len(deta))
    dphi  += [0] * (maxtracks - len(dphi))
    d0    += [0] * (maxtracks - len(d0  ))

    tracks = zip(imp, deta, dphi, d0)
    tracks = np.asarray(tracks)

    #print tracks

    return tracks

input_filename="inputs/2019-06-20/MC_abseta_0.0_1.3_et_0.0_500.0_processes_pid_01.h5"
print "input filename=",input_filename

#irec = int(sys.argv[1])

f = h5py.File(input_filename, 'r')
#retrieve dataset
dset = f['train']
print type(dset)
print dset["eventNumber"]
#print dset["eventNumber"].shape
#print dset.keys()

tmpMcChannels = list()


nEvents=len(dset["mcChannelNumber"])
#nCollect=43940 # 43945 this is the limit of type-1
nCollect=40000 # 43945 this is the limit of type-1
nCollected=0 #this is just a counter

# nEvents=100
# debug=True
debug=False

print "Looing over",nEvents,"events. Collecting", nCollect,"events of type",targetType

ZeeCount=0
#event_idx=0

#out_dir = "/eos/user/m/mociduki/el_images/"
#out_dir = "/afs/cern.ch/work/m/mociduki/el_images_tmpIFF/"
#out_dir = "/afs/cern.ch/work/m/mociduki/el_images_tmp_2class/"
#out_dir = "/afs/cern.ch/work/m/mociduki/el_images_tmp_tmp/"
#out_dir = "/afs/cern.ch/work/m/mociduki/el_images_tmp_0trk/"
#out_dir = "/afs/cern.ch/work/m/mociduki/el_images_tmp_1trk/"
#out_dir = "/afs/cern.ch/work/m/mociduki/el_images_tmp_1trk/"
out_dir = "/afs/cern.ch/work/m/mociduki/el_images_trk_40k/"
#out_dir = "/eos/user/m/mociduki/el_images_40k/"
#out_dir = "./"
out_name=out_dir+"electron_images_type%d.h5" % targetType
out_h5 = tables.open_file(out_name, mode='w')
grp = out_h5.create_group('/', 'data', 'yup')

print datetime.datetime.now()

print "entering event loop..."
for event_idx in range(nEvents):# dset["mcChannelNumber"]:

    if event_idx>0:
        if   event_idx < 1e2 and event_idx%10    ==0: print "nCollected / nProcessed=",nCollected,"/",event_idx,"=",nCollected/float(event_idx)
        elif event_idx < 1e3 and event_idx%100   ==0: print "nCollected / nProcessed=",nCollected,"/",event_idx,"=",nCollected/float(event_idx)
        elif event_idx < 1e4 and event_idx%1000  ==0: print "nCollected / nProcessed=",nCollected,"/",event_idx,"=",nCollected/float(event_idx)
        elif event_idx < 1e5 and event_idx%10000 ==0: print "nCollected / nProcessed=",nCollected,"/",event_idx,"=",nCollected/float(event_idx)
        elif event_idx < 1e6 and event_idx%100000==0: print "nCollected / nProcessed=",nCollected,"/",event_idx,"=",nCollected/float(event_idx)
        pass

    ### Read Variables ===============================================
    m_ee       = dset["ll_m"][event_idx]
    truthOrigin= dset["p_TruthOrigin"][event_idx]
    truthTypeRaw  = dset["p_TruthType"][event_idx]
    truthIFF      = dset["p_iffTruth"][event_idx]
    mcChannel  = dset["mcChannelNumber"][event_idx]

    # ### This is the real code ###
    # truthTypeMod=2                        #else than below
    # if    truthTypeRaw==2: truthTypeMod=0 #signal
    # elif  truthTypeRaw==4: truthTypeMod=1 #bkg electrons
    if nClasses==2:
        if    truthTypeRaw==2: truthTypeMod=0 #signal #truthIFF==2
        elif  truthTypeRaw==4: truthTypeMod=0 #signal #truthIFF==3
        else:                  truthTypeMod=1 #bkg electrons
    if nClasses==3:
        truthTypeMod=1                        #else than below
        if    truthTypeRaw==2: truthTypeMod=0 #type-2 electrons
        elif  truthTypeRaw==4: truthTypeMod=0 #type-4 electrons
    elif nClasses==5:
        # IFF definition : Our category
        truthTypeMod=5                    #else than below
        if   truthIFF== 0: truthTypeMod=4 # Unknown                
        elif truthIFF== 1: truthTypeMod=2 # KnownUnknown           
        elif truthIFF== 2: truthTypeMod=0 # IsoElectron            
        elif truthIFF== 3: truthTypeMod=1 # ChargeFlipIsoElectron  
        elif truthIFF== 4: truthTypeMod=4 # PromptMuon             
        elif truthIFF== 5: truthTypeMod=4 # PromptPhotonConversion 
        elif truthIFF== 6: truthTypeMod=4 # ElectronFromMuon       
        elif truthIFF== 7: truthTypeMod=3 # TauDecay               
        elif truthIFF== 8: truthTypeMod=3 # BHadronDecay           
        elif truthIFF== 9: truthTypeMod=3 # CHadronDecay           
        elif truthIFF==10: truthTypeMod=2 # LightFlavorDecay       
        else: print "WARNING: IFF truth label is out of range!!!"
        pass

    if debug or event_idx<10: print "mcChannel, truthTypeRaw, truthTypeMod=", mcChannel,truthTypeRaw, truthTypeMod

    if truthTypeMod!=targetType:
        event_idx+=1
        continue
        pass
        
    truthPt    = dset["p_truth_pt"][event_idx]
    truthPhi   = dset["p_truth_phi"][event_idx]
    truthEta   = dset["p_truth_eta"][event_idx]

    #print "Processing with TLorentzVector..."
    p_truth_4v= TLorentzVector()
    p_truth_4v.SetPtEtaPhiE(truthPt,truthEta,truthPhi,dset["p_truth_E"][event_idx])
    truthM     = p_truth_4v.M()

    #kinematics
    energy     = dset["p_e"][event_idx] #now in GeV
    pt_calo    = dset["p_et_calo"][event_idx]
    pt_track   = dset["p_pt_track"][event_idx]
    eta        = dset["p_eta"][event_idx]
    phi        = dset["p_phi"][event_idx]

    #PID variables
    Rhad       = dset["p_Rhad"][event_idx]
    f3         = dset["p_f3"][event_idx]
    weta2      = dset["p_weta2"][event_idx]

    Rphi       = dset["p_Rphi"][event_idx]
    Reta       = dset["p_Reta"][event_idx]
    Eratio     = dset["p_Eratio"][event_idx]

    f1         = dset["p_f1"][event_idx]
    d0         = dset["p_d0"][event_idx]    
    d0Sig      = dset["p_d0Sig"][event_idx]    

    dPoverP    = dset["p_dPOverP"][event_idx]    
    TRTPID     = dset["p_TRTPID"][event_idx]    
    deltaEta1  = dset["p_deltaEta1"][event_idx]    

    dPhiRes2   = dset["p_deltaPhiRescaled2"][event_idx]
    nSCT       = dset["p_numberOfSCTHits"][event_idx]

    if event_idx<100: 
        print "truth_pt=", truthPt, ", energy= ", energy, ", pt_trk=", pt_track,", pt_calo= ",pt_calo,"iffTruth= ", truthIFF, ", type=",truthTypeRaw

    # This classification is a bit ramdom atm
    llh_tight=0 # for signal
    if dset["p_LHTight"][event_idx]==0: 
        llh_tight=2 #background
        if dset["p_LHValue"][event_idx]<-1: llh_tight=1 #for fakes
        pass

    llh_medium=0 # for signal
    if dset["p_LHMedium"][event_idx]==0: 
        llh_medium=2 #background
        if dset["p_LHValue"][event_idx]<-1: llh_medium=1 #for fakes
        pass

    llh_loose=0 # for signal
    if dset["p_LHLoose"][event_idx]==0: 
        llh_loose=2 #background
        if dset["p_LHValue"][event_idx]<-1: llh_loose=1 #for fakes
        pass
        
    el_4v = TLorentzVector()
    el_4v.SetPtEtaPhiE(pt_calo,eta,phi,energy)

    # print "m_ee(Zee), truthOrig, truthType=",m_ee,truthOrigin,truthType
    # try: tmpMcChannels.index(mcNum)
    # except: tmpMcChannels.append(mcNum)

    # Read images and nomalize to the electron energy
    Lr0 = dset["em_barrel_Lr0"][event_idx].transpose()   / energy
    Lr1 = dset["em_barrel_Lr1"][event_idx].transpose()   / energy
    Lr2 = dset["em_barrel_Lr2"][event_idx].transpose()   / energy
    Lr3 = dset["em_barrel_Lr3"][event_idx].transpose()   / energy
    Lr4 = dset["tile_barrel_Lr1"][event_idx].transpose() / energy
    Lr5 = dset["tile_barrel_Lr2"][event_idx].transpose() / energy
    Lr6 = dset["tile_barrel_Lr3"][event_idx].transpose() / energy


    ### Fill Variables in the output file/group===============================================
    #table_name="table_%d" % event_idx
    table_name="table_%d" % nCollected
    out_h5.create_table(grp, table_name, Image)

    table = getattr(out_h5.root.data, table_name)
    image = table.row

    # s1 = tau_topo_image(index, rec, cal_layer=1, width=Image.columns['s1'].shape[1], height=Image.columns['s1'].shape[0])
    image['s0'] = Lr0
    image['s1'] = Lr1
    image['s2'] = Lr2
    image['s3'] = Lr3
    image['s4'] = Lr4
    image['s5'] = Lr5
    image['s6'] = Lr6

    image['tracks'] = tau_tracks_simple(dset, event_idx, el_4v)

    image['e'        ] = energy #rec['off_pt'] * np.cosh(rec['off_eta'])
    image['pt'       ] = pt_calo #rec['off_pt']
    image['eta'      ] = eta #rec['off_eta']
    image['phi'      ] = phi #rec['off_phi']

    image['mu'       ] = -1 #rec['averageintpercrossing']
    image['pantau'   ] = llh_tight #rec['off_decaymode']

    image['llh_tight'  ] = llh_tight
    image['llh_medium' ] = llh_medium
    image['llh_loose'  ] = llh_loose

    image['truthmode'] = truthTypeMod #rec['true_decaymode']
    image['true_pt'  ] = truthPt #rec['true_pt']
    image['true_eta' ] = truthEta #rec['true_eta']
    image['true_phi' ] = truthPhi #rec['true_phi']
    image['true_m'   ] = truthM if not math.isinf(truthM) else -1 #rec['true_m']
    #image['alpha_e'  ] = truthPt / pt_calo * np.cosh(eta)

    image['Rhad'    ]   = Rhad    
    image['f3'      ]   = f3      
    image['weta2'   ]   = weta2   

    image['Rphi'    ]   = Rphi    
    image['Reta'    ]   = Reta    
    image['Eratio'  ]   = Eratio  

    image['f1'      ]   = f1      
    image['d0'      ]   = d0      
    image['d0Sig'   ]   = d0Sig   

    image['dPoverP' ]   = dPoverP   
    image['TRTPID'  ]   = TRTPID  
    image['deltaEta1']  = deltaEta1

    image['dPhiRes2']   = dPhiRes2
    image['nSCT'    ]   = nSCT    

    image.append()
    nCollected+=1

    if nCollected==nCollect: 
        print "Collected enough, breaking loop..."
        break
        pass

    event_idx+=1
    pass

print "Collected",nCollected,"events in the end."
print datetime.datetime.now()


