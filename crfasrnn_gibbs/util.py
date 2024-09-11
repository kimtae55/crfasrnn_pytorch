"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import numpy as np
import torch
from scipy.interpolate import UnivariateSpline

def p_lis(gamma_1, threshold=0.1, label=None, savepath=None):
    '''
    Rejection of null hypothesis are shown as 1, consistent with online BH, Q-value, smoothFDR methods.
    # LIS = P(theta = 0 | x)
    # gamma_1 = P(theta = 1 | x) = 1 - LIS
    '''
    gamma_1 = gamma_1.ravel()
    dtype = [('index', int), ('value', float)]
    size = gamma_1.shape[0]

    # flip
    lis = np.zeros(size, dtype=dtype)
    lis[:]['index'] = np.arange(0, size)
    lis[:]['value'] = 1-gamma_1 

    # get k
    lis = np.sort(lis, order='value')
    cumulative_sum = np.cumsum(lis[:]['value'])
    k = np.argmax(cumulative_sum > (np.arange(len(lis)) + 1)*threshold)

    signal_lis = np.zeros(size)
    signal_lis[lis[:k]['index']] = 1

    if savepath is not None:
        np.save(os.path.join(savepath, 'gamma.npy'), gamma_1)
        np.save(os.path.join(savepath, 'lis.npy'), signal_lis)

    if label is not None:
        # GT FDP
        rx = k
        sigx = np.sum(1-label[lis[:k]['index']])
        fdr = sigx / rx if rx > 0 else 0

        # GT FNR
        rx = size - k
        sigx = np.sum(label[lis[k:]['index']]) 
        fnr = sigx / rx if rx > 0 else 0

        # GT ATP
        atp = np.sum(label[lis[:k]['index']]) 
        return fdr, fnr, atp

def compute_kl_divergence(p, q):
    kl_div = np.sum((p+1e-8) * np.log((p+1e-8) / (q.detach().numpy()+1e-8)))
    print('compute_kl_divergence() > kl_div: ', kl_div)
    return kl_div

def qvalue(pvals, threshold=0.05, verbose=False):
    """Function for estimating q-values from p-values using the Storey-
    Tibshirani q-value method (2003).

    Input arguments:
    ================
    pvals       - P-values corresponding to a family of hypotheses.
    threshold   - Threshold for deciding which q-values are significant.

    Output arguments:
    =================
    significant - An array of flags indicating which p-values are significant.
    qvals       - Q-values corresponding to the p-values.
    """

    """Count the p-values. Find indices for sorting the p-values into
    ascending order and for reversing the order back to original."""
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    rev_ind = np.argsort(ind)
    pvals = pvals[ind]

    # Estimate proportion of features that are truly null.
    kappa = np.arange(0, 0.96, 0.01)
    pik = [sum(pvals > k) / (m*(1-k)) for k in kappa]
    cs = UnivariateSpline(kappa, pik, k=3, s=None, ext=0)
    pi0 = float(cs(1.))
    if (verbose):
        print('The estimated proportion of truly null features is %.3f' % pi0)

    """The smoothing step can sometimes converge outside the interval [0, 1].
    This was noted in the published literature at least by Reiss and
    colleagues [4]. There are at least two approaches one could use to
    attempt to fix the issue:
    (1) Set the estimate to 1 if it is outside the interval, which is the
        assumption in the classic FDR method.
    (2) Assume that if pi0 > 1, it was overestimated, and if pi0 < 0, it
        was underestimated. Set to 0 or 1 depending on which case occurs.

    I'm choosing second option 
    """
    if pi0 < 0:
        pi0 = 0
    elif pi0 > 1:
        pi0 = 1

    # Compute the q-values.
    qvals = np.zeros(np.shape(pvals))
    qvals[-1] = pi0*pvals[-1]
    for i in np.arange(m-2, -1, -1):
        qvals[i] = min(pi0*m*pvals[i]/float(i+1), qvals[i+1])

    # Test which p-values are significant.
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind] = qvals<threshold

    """Order the q-values according to the original order of the p-values."""
    qvals = qvals[rev_ind]
    return significant, qvals
'''
import plotly.graph_objs as go
def visualize_3d_mesh(data):
    # Create a meshgrid of coordinates
    x, y, z = np.meshgrid(np.arange(data.shape[0]),
                          np.arange(data.shape[1]),
                          np.arange(data.shape[2]))

    # Flatten the data and magnitudes arrays
    data_flat = data.flatten()
    magnitudes = np.abs(data_flat)

    # Create a Scatter3d trace with size based on magnitude
    scatter = go.Scatter3d(x=x.flatten(),
                           y=y.flatten(),
                           z=z.flatten(),
                           mode='markers',
                           marker=dict(size=magnitudes * 10,  # Adjust the multiplier for appropriate size scaling
                                       color='blue',  # You can set the color as needed
                                       opacity=0.8)
                           )

    # Set layout properties
    layout = go.Layout(scene=dict(aspectmode='cube'))

    # Create a figure and add the scatter trace
    fig = go.Figure(data=[scatter], layout=layout)

    # Show the interactive plot
    fig.show()

'''