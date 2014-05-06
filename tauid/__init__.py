import os
import logging
from variables import *

log = logging.getLogger('tauid')
if not os.environ.get('DEBUG', False):
    log.setLevel(logging.INFO)



VARIABLES = {
    'plotting': [pt, eta, npv, mu],
    'plotting_id': [centfrac,
                    pssfraction,
                    nstrip,
                    emradius,
                    hadradius,
                    emfraction,
                    hadfraction,
                    stripwidth,
                    lead2clustereoverallclusterE,
                    lead3clustereoverallclusterE,
                    numefftopoclusters,
                    efftopoinvmass,
                    efftopomeandeltar,
                    numtopoclusters,
                    topoinvmass,
                    topomeandeltar,
                    nwidetrk,
                    trkavgdist,
                    chpiemeovercaloeme,
                    etoverptleadtrk,
                    empovertrksysp,
                    ipsigleadtrk,
                    drmax,
                    masstrksys,
                    trflightpathsig,
                    ],
    'presel_1': [centfrac,
                 pssfraction,
                 nstrip,
                 emradius,
                 hadradius,
                 emfraction,
                 hadenergy,
                 stripwidth,
                 lead2clustereoverallclusterE,
                 lead3clustereoverallclusterE,
                 numtopoclusters,
                 topoinvmass,
                 topomeandeltar,
                 ],
    'presel_2': [centfrac,
                 pssfraction,
                 nstrip,
                 emradius,
                 hadradius,
                 emfraction,
                 hadenergy,
                 stripwidth,
                 lead2clustereoverallclusterE,
                 lead3clustereoverallclusterE,
                 numefftopoclusters,
                 efftopoinvmass,
                 efftopomeandeltar,
                 ],
    'presel_3': [centfrac,
                 pssfraction,
                 nstrip,
                 emradius,
                 hadradius,
                 emfraction,
                 hadfraction,
                 stripwidth,
                 lead2clustereoverallclusterE,
                 lead3clustereoverallclusterE,
                 numefftopoclusters,
                 efftopoinvmass,
                 efftopomeandeltar,
                 ],
    'presel_3var': [centfrac,
                    pssfraction,
                    nstrip,
                    ],
    'presel_5var': [centfrac,
                    pssfraction,
                    nstrip,
                    emradius,
                    hadradius,
                    ],
    'full_1p': [centfrac,
                pssfraction,
                nstrip,
                nwidetrk,
                trkavgdist,
                chpiemeovercaloeme,
                etoverptleadtrk,
                empovertrksysp,
                ipsigleadtrk,
                ],
    'full_mp': [centfrac,
                pssfraction,
                nstrip,
                nwidetrk,
                trkavgdist,
                chpiemeovercaloeme,
                etoverptleadtrk,
                empovertrksysp,
                drmax,
                masstrksys,
                trflightpathsig,
                ],
    }
