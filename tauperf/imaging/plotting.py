import math
import numpy as np
import itertools
import socket
import matplotlib as mpl;

techlab_hosts = [
    'techlab-gpu-nvidiak20-03.cern.ch',
    'techlab-gpu-nvidiagtx1080-07.cern.ch',
    'tlab-gpu-gtx1080ti-07.cern.ch',
]
if socket.gethostname() in techlab_hosts:
    mpl.use('PS')
else:
    mpl.use('PS')
#    mpl.use('TkAgg')


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LogNorm

from . import log; log = log.getChild(__name__)

def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi

def get_wp(true_pos, false_pos, thresh, method='corner', target_value=0.8):
    if (true_pos.ndim, false_pos.ndim, thresh.ndim) != (1, 1, 1):
        raise ValueError('wrong dimension')
    if len(true_pos) != len(false_pos) or len(true_pos) != len(thresh):
        raise ValueError('wrong size')


    if method == 'corner':
        # compute the distance to the (0, 1) point
        dr_square = true_pos * true_pos + (false_pos - 1) * (false_pos - 1)
        # get the index in the dr_square array
        index_min = np.argmin(dr_square)
        # return optimal true positive eff, false positive eff and threshold for cut
    elif method == 'target_eff':
        val = target_value
        target_eff = np.abs(true_pos - val)
        index_min = np.argmin(target_eff)
    elif method == 'target_rej':
        target_rej = np.abs(false_pos - val)
        index_min = np.argmin(target_rej)
    else:
        raise ValueError('wrong method argument')

    return true_pos[index_min], false_pos[index_min], thresh[index_min]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          name='plots/imaging/confusion_matrix.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    np.set_printoptions(precision=2)
    diagonal = float(np.trace(cm)) / float(np.sum(cm))
    log.info('Diag / Total = {0} / {1} = {2}%'.format(np.trace(cm), np.sum(cm), 100 * diagonal))
    cm = cm.T.astype('float') / cm.T.sum(axis=0)
    cm = np.rot90(cm.T, 1)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes[::-1])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)#[:, np.newaxis]

    plt.colorbar()

    log.info('')
    for l in cm:
        log.info('\t {}'.format(l))
    log.info('')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:1.2f}'.format(100 * cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > 100 * thresh else "black")

    plt.xlabel('True Tau Decay Mode')
    plt.ylabel('Reconstructed Tau Decay Mode')
    plt.tight_layout()
    plt.savefig(name)


# pt efficiency
def calc_eff(accept, total, binning=(20, 0, 100), name='1p1n'):
    from root_numpy import fill_hist
    from rootpy.plotting import Hist, Efficiency
    hnum = Hist(binning[0], binning[1], binning[2])
    hden = Hist(binning[0], binning[1], binning[2])
    fill_hist(hnum, accept)
    fill_hist(hden, total)
    eff = Efficiency(hnum, hden, name='eff_' + name, title=name)
    return eff

def get_eff(arr, pred, scale=1., binning=(20, 0, 100), color='red', name='1p1n'):
    total = arr * scale
    accept = total[pred == 1]
    eff = calc_eff(accept, total, binning=binning, name=name)
    eff.color = color
    return eff


def plot_image(rec, eta, phi, ene, irec, cal_layer, suffix):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    rect = mpatches.Rectangle(
        (-0.2, -0.2), 0.4, 0.4, 
        fill=False, linewidth=3, label='selection')
    plt.scatter(
        eta, phi, c=ene, marker='s', s=40,
        label= 'Number of cells = {0}'.format(len(eta)))
    plt.colorbar()
    ax.add_patch(rect)
    plt.plot(
        rec['true_charged_eta'] - rec['true_eta'], 
        dphi(rec['true_charged_phi'], rec['true_phi']), 'ro', 
        label='charge pi, pT = %1.2f GeV' % (rec['true_charged_pt'] / 1000.))
    if not '0n' in suffix:
        plt.plot(
            rec['true_neutral_eta'] - rec['true_eta'], 
            dphi(rec['true_neutral_phi'], rec['true_phi']), 'g^', 
            label='neutral pi, pT = %1.2f GeV' % (rec['true_neutral_pt'] / 1000.))
    plt.xlim(-0.4, 0.4)
    plt.ylim(-0.4, 0.4)
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.title('{0}: image {1} sampling {2}'.format(suffix, irec, cal_layer))
    plt.legend(loc='upper right', fontsize='small', numpoints=1)
    fig.savefig('plots/imaging/images/image_{0}_s{1}_{2}.pdf'.format(
            irec, cal_layer, suffix))
    fig.clf()
    plt.close()

def plot_heatmap(image, rec, pos_central_cell, irec, cal_layer, suffix, fixed_scale=False):
    fig = plt.figure()
    if fixed_scale:
        image[image <= 0] = 0.00001

    plt.imshow(
        image, 
        extent=[-0.2, 0.2, -0.2, 0.2], 
        interpolation='nearest',  
        cmap=plt.cm.Reds if fixed_scale else plt.cm.viridis,
        norm=None if fixed_scale is False else LogNorm(0.0001, 1))

    plt.colorbar()
    plt.plot(
        rec['true_charged_eta'] - rec['true_eta'], 
        dphi(rec['true_charged_phi'], rec['true_phi']), 'ro', 
        label='true charge pi, pT = %1.2f GeV' % (rec['true_charged_pt'] / 1000.))

    plt.plot(
        rec['off_tracks_eta'][0] - pos_central_cell['eta'], 
        dphi(rec['off_tracks_phi'][0], pos_central_cell['phi']), 'bo', 
        label='reco charge pi, pT = %1.2f GeV' % (rec['off_tracks_pt'][0] / 1000.))


    if not '0n' in suffix:
        plt.plot(
            rec['true_neutral_eta'] - rec['true_eta'], 
            dphi(rec['true_neutral_phi'], rec['true_phi']), 'g^', 
            label='true neutral pi, pT = %1.2f GeV' % (rec['true_neutral_pt'] / 1000.))
    plt.xlabel('eta')
    plt.ylabel('phi')
    plt.title('{0}: image {1} sampling {2}'.format(suffix, irec, cal_layer))
    plt.legend(loc='upper right', fontsize='small', numpoints=1)
    fig.savefig('plots/imaging/images/image_{0}_s{1}_{2}.pdf'.format(
            irec, cal_layer, suffix))
    fig.clf()
    plt.close()

def divide_0(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

nClass=2
def plot_roc(y_test, y_pred, y_pant):
    from sklearn.metrics import roc_curve

    y_test_1p = y_test[np.logical_or(y_test == 0, y_test == 1, y_test == 2)]
    y_pred_1p = y_pred[np.logical_or(y_test == 0, y_test == 1, y_test == 2)]
    denom=y_pred_1p[:,0] + y_pred_1p[:,1]
    if nClass>=3: denom=y_pred_1p[:,0] + y_pred_1p[:,1] + y_pred_1p[:,2]
    y_pred_1p = divide_0(y_pred_1p[:,0], denom)
    fpr_1p0n, tpr_1p0n, _ = roc_curve(y_test_1p, y_pred_1p, pos_label=0)

    y_pant_1p0n = y_pant[y_test == 0]
    y_pant_1pXn = y_pant[np.logical_or(y_test == 1, y_test == 2)]
    eff_pant_1p0n = float(len(y_pant_1p0n[y_pant_1p0n == 0])) / float(len(y_pant_1p0n))
    rej_pant_1p0n = float(len(y_pant_1pXn[y_pant_1pXn != 0])) / float(len(y_pant_1pXn))
    
    y_test_1pXn = y_test[np.logical_or(y_test == 1, y_test == 2)]
    y_pred_1pXn = y_pred[np.logical_or(y_test == 1, y_test == 2)]
    denom=y_pred_1pXn[:,0] + y_pred_1pXn[:,1]
    if nClass>=3: denom=y_pred_1pXn[:,1] + y_pred_1pXn[:,2]
    y_pred_1pXn = divide_0(y_pred_1pXn[:,1], denom)
    fpr_1p1n, tpr_1p1n, _ = roc_curve(y_test_1pXn, y_pred_1pXn, pos_label=1)

    y_pant_1p1n = y_pant[y_test == 1]
    y_pant_1p2n = y_pant[y_test == 2]
    eff_pant_1p1n = float(len(y_pant_1p1n[y_pant_1p1n == 1])) / float(len(y_pant_1p1n))
    rej_pant_1p1n = 0 #float(len(y_pant_1p2n[y_pant_1p2n != 1])) / float(len(y_pant_1p2n))

    if nClass==5: 
        y_test_3p = y_test[np.logical_or(y_test == 3, y_test == 4)]
        y_pred_3p = y_pred[np.logical_or(y_test == 3, y_test == 4)]
        y_pant_3p = y_pant[np.logical_or(y_test == 3, y_test == 4)]

        y_pred_3p = divide_0(y_pred_3p[:,3], (y_pred_3p[:,3] + y_pred_3p[:,4]))
        fpr_3p0n, tpr_3p0n, _ = roc_curve(y_test_3p, y_pred_3p, pos_label=3)
        
        y_pant_3p0n = y_pant[y_test == 3]
        y_pant_3pXn = y_pant[y_test == 4]
        eff_pant_3p0n = float(len(y_pant_3p0n[y_pant_3p0n == 3])) / float(len(y_pant_3p0n))
        rej_pant_3p0n = float(len(y_pant_3pXn[y_pant_3pXn != 3])) / float(len(y_pant_3pXn))
        pass
    
    plt.figure()
    plt.plot(tpr_1p0n, 1 - fpr_1p0n, label='sig vs fake+bkg', color='red')
    plt.plot(tpr_1p1n, 1 - fpr_1p1n, label='fake vs sig+bkg', color='blue')
    if nClass==5: 
        plt.plot(tpr_3p0n, 1 - fpr_3p0n, label='3p0n vs 3pXn', color='purple')
        pass
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])

    plt.scatter(
        [eff_pant_1p0n, eff_pant_1p1n],
        [rej_pant_1p0n, rej_pant_1p1n],
        s=100,
        marker='v',
        c=['red', 'blue'],
        label='LLH-tight')
    if nClass==5: 
        plt.scatter(
            [eff_pant_1p0n, eff_pant_1p1n, eff_pant_3p0n],
            [rej_pant_1p0n, rej_pant_1p1n, rej_pant_3p0n],
            s=100,
            marker='v',
            c=['red', 'blue', 'purple'],
            label='LLH-tight')
    
    axes = plt.gca()
    axes.xaxis.set_ticks(np.arange(0, 1, 0.1))
    axes.yaxis.set_ticks(np.arange(0, 1, 0.1))
    axes.grid(True)
    
    plt.legend(loc='lower left', fontsize='small', numpoints=3)
    plt.savefig('./plots/imaging/roc_curve.pdf')


def plot_scores(y_pred, y_true):
    
    nCateg = int(2)
    if   nClass==3: nCateg = 3
    elif nClass==5: nCateg = 5
    for i in xrange(nCateg):
        y = y_pred[:,i]
        fig = plt.figure()
        plt.hist([
                y[y_true == 0], 
                y[y_true == 1] ],
                 #label=['1p0n', '1p1n', '1p2n'],
                 label=['signal', 'others'],
                 bins=20, range=(0, 1), stacked=True, log=True)
        if nClass==3:
            plt.hist([
                y[y_true == 0], 
                y[y_true == 1], 
                y[y_true == 2]],
                     #label=['1p0n', '1p1n', '1p2n'],
                     label=['signal', 'non-prompt', 'others'],
                    bins=20, range=(0, 1), stacked=True, log=True)
        elif nClass==5:
            plt.hist([
                y[y_true == 0], 
                y[y_true == 1], 
                y[y_true == 2], 
                y[y_true == 3], 
                y[y_true == 4]],
                     label=['1p0n', '1p1n', '1p2n', '3p0n', '3p1n'],
                     bins=20, range=(0, 1), stacked=True, log=True)
            pass
        plt.legend(fontsize='small', numpoints=3)
        plt.xlabel('classifier score {0}'.format(i))
        plt.ylabel('Number of Events')
        fig.savefig('./plots/imaging/scores_{0}_stacked.pdf'.format(i))

    max_score = np.max(y_pred, axis=1)
    fig = plt.figure()
    plt.hist([
            max_score[y_true == 0], 
            max_score[y_true == 1], 
            max_score[y_true == 2], 
            max_score[y_true == 3], 
            max_score[y_true == 4]],
             label=['1p0n', '1p1n', '1p2n', '3p0n', '3p1n'],
             bins=20, range=(0, 1), stacked=True, log=True)
    plt.legend(fontsize='small', numpoints=3)
    plt.xlabel('maximum score'.format(i))
    plt.ylabel('Number of Events')
    fig.savefig('./plots/imaging/scores_max_stacked.pdf'.format(i))
    
    

def plot_kinematics(test, pt_pred, eta_pred, phi_pred, m_pred):
    fig = plt.figure()
    plt.hist([test['true_pt'] / 1000., pt_pred], bins=20, range=(0., 200.))
    fig.savefig('./plots/imaging/pt.pdf')

    fig = plt.figure()
    plt.hist([test['true_eta'], eta_pred], bins=20, range=(-1, 1))
    fig.savefig('./plots/imaging/eta.pdf')

    fig = plt.figure()
    plt.hist([test['true_phi'], phi_pred], bins=20, range=(-1, 1))
    fig.savefig('./plots/imaging/phi.pdf')

    fig = plt.figure()
    plt.hist([test['true_m'] / 1000., m_pred], bins=20, range=(0, 2))
    fig.savefig('./plots/imaging/mass.pdf')


def plot_reg(pred, true, calo, ene):
    
    reg = pred[:,0] * ene

    fig = plt.figure()
    plt.hist([reg, true, calo],
              label=['reg', 'truth', 'calo'],
              bins=20, range=(0, 100000), stacked=False, log=True)
    plt.legend(fontsize='small', numpoints=3)
    plt.xlabel('target variable')
    plt.ylabel('Number of Events')
    fig.savefig('./plots/imaging/target.pdf')

    fig = plt.figure()
    diff_pred = (reg - true) / true
    diff_calo = (calo - true) / true
    plt.hist([diff_pred, diff_calo],
             label=['reg', 'calo'],
             bins=40, range=(-1, 1), stacked=False)
    plt.legend(fontsize='small', numpoints=3)
    plt.xlabel('(estimator - truth) / truth')
    plt.ylabel('Number of Events')
    fig.savefig('./plots/imaging/target_diff.pdf')

    plt.yscale('log')
    fig.savefig('./plots/imaging/target_diff_log.pdf')

#     print np.mean(diff_pred), np.mean(diff_calo)
#     print np.quantile(np.abs(diff_pred), 0.68), np.quantile(np.abs(diff_calo), 0.68)
#     arr = np.array([true, diff_pred, diff_calo]).T
#     print arr[0]
#     arr = np.sort(arr, axis=0)
#     arr = np.array_split(arr, 20)
    
#     x = np.array([
#             np.mean(a[:,0]) / 1000. for a in arr])

#     mean_calo = np.array([
#             np.mean(a[:,2]) for a in arr])
#     mean_pred = np.array([
#             np.mean(a[:,1]) for a in arr])

#     res_68_calo = np.array([
#             np.quantile(np.abs(a[:,2]), 0.68) for a in arr])
#     res_68_pred = np.array([
#             np.quantile(np.abs(a[:,1]), 0.68) for a in arr])

#     fig = plt.figure()
#     plt.plot(x, res_68_calo, 'bs', label='calo')
#     plt.plot(x, res_68_pred, 'g^', label='regression')
#     plt.legend(fontsize='small', numpoints=3)
#     plt.xlim(0, 200)
#     plt.xlabel('True transverse momentum [GeV]')
#     plt.ylabel('Relative resolution')
#     fig.savefig('./plots/imaging/resolution.pdf')
#     fig = plt.figure()

#     plt.plot(x, mean_calo, 'bs', label='calo')
#     plt.plot(x, mean_pred, 'g^', label='regression')
#     plt.legend(fontsize='small', numpoints=3)
#     plt.xlim(0, 200)
#     plt.xlabel('True transverse momentum [GeV]')
#     plt.ylabel('< estimate - true / true >')
#     fig.savefig('./plots/imaging/mean.pdf')

#     print 'pt   |\t mean \t| mean (corr) \t | quantile \t | quantile (cor)'
#     for a in arr:
#         pt = np.mean(a[:,0]) / 1000.
#         mean = np.mean(a[:,1])
#         mean_corr = np.mean(a[:,1] - mean)
#         quantile = np.quantile(np.abs(a[:,1]), 0.68)
#         quantile_corr = np.quantile(np.abs(a[:,1] -  np.mean(a[:,1])), 0.68)
#         print '{0:1.3f} | {1:1.3f} | {2:1.3f} | {3:1.3f} | {4:1.3f}'.format(
#             pt, mean, mean_corr, quantile, quantile_corr)
