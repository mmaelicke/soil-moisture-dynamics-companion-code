import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist
from shapely import wkt
import skgstat as skg
from sklearn.cluster import MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.isotonic import IsotonicRegression
import skinfo
import itertools


minmax = lambda x:MinMaxScaler().fit_transform(x)

# extract a new period
def extract(list_of_df, start, stop):
    for df in list_of_df:
        yield df[start:stop]
        
def dispersion(x):
    return (1 / len(x)) * np.sum(x**4)

# plot the overview moisture and rank plots
def plot_overview(data):
    fig, axes = plt.subplots(2 * len(data),1, figsize=(12, len(data) * 2 * 4))
    
    # moisture plots
    for i in range(int(len(axes) / 2)):
        data[i].plot(ax=axes[i], legend=False, style='-b')
        axes[i].set_ylabel('soil moisture [$\Theta$]', usetex=True)
        
    for i in range(int(len(axes) / 2), len(axes)):
        data[int(i - (len(axes)/2))].rank(axis=1, pct=True).plot(ax=axes[i], legend=False)
        axes[i].set_ylabel('rel. rank [-]')
    return fig

def plot_ranks(data, ax1, ax2, use_mean=False):
    # plot data
    data.plot(ax=ax1, legend=False, style='-b', sharex=True)
    ax1.set_ylabel('soil moisture [$\Theta$]', usetex=False)
    
    # plot ranks
    ranks = data.rank(axis=1, pct=True)
    cmp = plt.get_cmap('seismic_r')
    plt.xticks(rotation=45)
    if use_mean:
        index = ranks.mean().sort_values().index
    else:
        index = ranks.iloc[0].sort_values().index
    for i, col in enumerate(index):
        arr = ranks[col].values
        ax2.plot(ranks.index, arr, linestyle='-', color=cmp(i / len(ranks.columns)), marker='.')

    ax2.set_ylabel('relative rank [-]')
    
    return ax2.get_figure()


def variograms(data, geometries, window=7, N=10, estimator='matheron', maxlag='median', binify='uniform',
               cm=plt.cm.Reds, styles=['-b', '--k', ':k'], rank=True):
    assert len(data) == len(geometries)
   
    # override skgstat's entropy method for using global bins
    if estimator == 'entropy':
        estimator = entropy_f
    if estimator == 'dispersion':
        estimator = dispersion
        
    n = len(data)
    v = [list() for _ in range(n)]
    
    for j, (df, geom) in enumerate(zip(data, geometries)):
        # variogrmas
        for i in range(0, len(df.index) - window, 1):
            if rank:
                df = df.rank(axis=1, pct=True)
            df_window = df.iloc[i:i+window].mean().dropna()
            #c = geom.loc[values.index].values
            c = geom.reindex(df_window.index).dropna()
            values = df_window.reindex(c.index).dropna()
            V = skg.Variogram(coordinates=c.values, values=values.values, n_lags=N, estimator=estimator, 
                              maxlag=maxlag, bin_func=binify)
            v[j].append(V)
    return v

def plot_variogram(list_of_variograms, ax, cm=plt.cm.Reds, xlabel=True, ylabel=True, norm=False):
    ma = np.max([np.max(_.experimental) for _ in list_of_variograms])
    for i,v in enumerate(list_of_variograms):
        if i == 0:
            bins = v.bins
        y = v.experimental
        if norm:
            y = [_ / ma for _ in y]
        ax.plot(bins, y, linestyle='-', color=cm(i / len(list_of_variograms)))
        if xlabel:
            ax.set_xlabel('lag [m]')
        if ylabel:
            if v.estimator.__name__ == 'entropy_f':
                ax.set_ylabel('Entropy [bit]')
            else:
                ax.set_ylabel('dispersion [-]')


def cluster_variograms(variograms, bandwidth=None):
    mean_shifts = list()
    # for each depth
    for v in variograms:        
        v_ = np.asarray([_v.experimental for _v in v])
        bw = np.percentile(pdist(v_), bandwidth) if bandwidth is not None else None
        ms = MeanShift(bandwidth=bw)
        ms.fit(v_)
        mean_shifts.append(ms)
    return mean_shifts


def plot_cluster(list_of_variograms, mean_shift, ax, cmap='gist_ncar', alpha=0.4, relative=False, xlabel=True, ylabel=True, norm=False):
    cmap = cm.get_cmap(cmap)
    colors = itertools.cycle(['blue', 'orange', 'darkgreen', 'gray', 'purple', 'black', 'pink'])
    # get the number of cluster
    N = len(mean_shift.cluster_centers_)
    ma = np.max([np.max(_.experimental) for _ in list_of_variograms])
    
    # get the bins
    x = range(len(list_of_variograms[0].bins)) if relative else list_of_variograms[0].bins
    
    # plot each cluster and all cluster members
    for center in range(N):
        _col = next(colors)
        y = np.asarray(list_of_variograms)[np.where(mean_shift.labels_==center)]
        center_vals = mean_shift.cluster_centers_[center]
        if norm:
            center_vals = [_ / ma for _ in center_vals]
        
        for _y in y:
            vals = _y.experimental
            if norm:
                vals = [_ / ma for _ in vals]
            ax.plot(x, vals, linestyle=':', alpha=alpha, color=_col)
        
        ax.plot(x, center_vals, linestyle='-', color=_col, lw=3, alpha=1)
        ax.annotate('%d clusters' % N, xy=(0.1, 0.75), xycoords='axes fraction')
        if xlabel:
            ax.set_xlabel('lag [%s]' % ('m / max(m)' if relative else 'm'))
        if ylabel:
            ax.set_ylabel('dispersion (%s)' % (list_of_variograms[0].estimator.__name__))


def compress_cluster(mean_shift, variograms, normalize=True):
    monos = []
    x_bins = variograms[0].bins
    for centroid in mean_shift.cluster_centers_:
        mono = IsotonicRegression().fit_transform(x_bins, centroid)
        if normalize:
            ma = np.nanmax([np.nanmax(_.experimental) for _ in variograms])
            mono = np.asarray([_ / ma for _ in mono])
        monos.append(mono)
    return x_bins, monos
    

def plot_compressed(x_bins, compressed, ax, xlabel=True, ylabel=True):
    colors = itertools.cycle(['blue', 'orange', 'darkgreen', 'gray', 'purple', 'black', 'pink'])
    for centroid in compressed:
        ax.plot(x_bins, centroid, color=next(colors), lw=5)
    if xlabel:
        ax.set_xlabel('lag [m]')
    if ylabel:
        ax.set_ylabel('dispersion [-]')
    
    return ax
    
    
def heat_diagram(variograms, clusters, figsize=None):
    #cmaps = [cm.get_cmap(name) for name in ('Blues','Oranges', 'Greens', 'Greys', 'Purples', 'bone_r', 'pink_r')]
    # create the figure
    n = len(variograms)
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    if figsize is None:
        figsize = (cols*5, rows*8)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True)
        
    for j, tup in enumerate(zip(variograms, clusters)):
        v, c = tup
        v = [_.experimental for _ in v]
        c = c.labels_
        # normalize the variograms
        normv = [minmax(_v.reshape(-1,1)).flatten() for _v in v]
        _normv = np.asarray([interp1d(range(_v.shape[0]), _v)(np.linspace(0, _v.shape[0] - 1, _v.shape[0] * 10)) for _v in normv])
        
        c_idx = np.unique(c)
        c_mask = np.column_stack((c,) * _normv.shape[1])
        
        plt.xticks(rotation=45)
        
        colors = itertools.cycle(('Blues','Oranges', 'Greens', 'Greys', 'Purples', 'bone_r', 'pink_r'))
        cmaps = [cm.get_cmap(next(colors)) for _ in range(len(c_idx))]
            
        for i in c_idx:
            axes.flatten()[j].matshow(np.where(c_mask==i, _normv, np.nan), cmap=cmaps[i])
            axes.flatten()[j].set_xticklabels(['', '0%', '20%', '40%', '60%', '80%', '100%'])
            axes.flatten()[j].set_xlabel('lag')
            if i == 0 and j == 0:
                axes.flatten()[j].set_ylabel('time [days]')
            axes.flatten()[j].xaxis.tick_bottom()
    
    return fig

def clustered_series(data, mean_shift, ax, xlabel=False, ylabel=False, rainfall=None, cumsum=False, temperature=None, legend=True, cl_threshold=10, bbox=(0, 0), show_spines=True):
    colors = itertools.cycle(['blue', 'orange', 'darkgreen', 'gray', 'purple', 'black', 'pink'])
    elements = []
    
    # plot the clusters
    for cl in np.unique(mean_shift.labels_):
        d = data.iloc[np.where(mean_shift.labels_ == cl)]
        x = d.index.to_pydatetime()
        y = d.values
        c = next(colors)
        ax.plot(x, y, marker='.', linestyle='', color=c)
        #data.iloc[np.where(mean_shift.labels_ == cl)].plot(ax=ax, marker='.', linestyle='', legend=None, color=next(colors))
        ax.set_ylabel('soil water content [cm³/cm³]')
        
        if legend and len(d.index) > cl_threshold:
            elements.append(Line2D([0], [0], color=c, lw=3, label='Cluster #%d' % (cl + 1)))
            
    use_cumsum = False
        
    # put in rainfall
    if rainfall is not None:
        ax.spines['right'].set_visible(show_spines)
        ax.set_ylim((0, 0.8))
        ax2 = ax.twinx()
        ax2.bar(rainfall.index.to_pydatetime(), rainfall.values, color='b')                    
        lim = ax2.get_ylim()
        ax2.set_ylim((rainfall.max() * 1.6, 0))  
        if show_spines:     
            ax2.set_ylabel('rainfall [mm]')
        else:
            ax2.set_yticks([])

        if cumsum:
            use_cumsum = True
            ax3 = ax.twinx()
            if show_spines:
                ax3.spines['right'].set_position(("axes", 1.08))
                ax3.spines['right'].set_visible(True)
            rr = rainfall.cumsum()
            ax3.plot(rr.index.to_pydatetime(), rr.values, color='white', alpha=0.8, lw=5)
            ax3.plot(rr.index.to_pydatetime(), rr.values, color='b', lw=3)
            if show_spines:
                ax3.set_ylabel('cum. rainfall [mm]')
            else:
                ax3.set_yticks([])
            if legend:
                elements.append(Line2D([0], [0], color='b', lw=3, label='cum. rainfall'))

        
    if temperature is not None:
        if show_spines:
            ax.spines['right'].set_visible(True)
        tax = ax.twinx()
        if rainfall is not None:
            d = 1.16 if use_cumsum else 1.08
        else: 
            d = 1.0
        if show_spines:
            tax.spines['right'].set_position(("axes", d))
            tax.spines['right'].set_visible(True)
        
        dd = temperature.where(temperature > 5).fillna(value=0).cumsum()
        tax.plot(dd.index.to_pydatetime(), dd.values, color='white', alpha=0.8, lw=5)
        tax.plot(dd.index.to_pydatetime(), dd.values, color='r', lw=3)
        
        if show_spines:
            lim = tax.get_ylim()
            tax.set_ylim(0, float(dd.max()))
            tax.set_ylabel('cum. day-degree [Kd]')
        else:
            tax.set_yticks([])
        
        if legend:
            elements.append(Line2D([0], [0], color='r', lw=3, label='day-degree'))
        
    if legend:
        ax.legend(handles=elements, loc='center right', bbox_to_anchor=bbox)
    
    return ax


def variogram_entropy(x, bins):
    count = np.histogram(x, bins=bins)[0]
    p = (count / np.sum(count)) + 1e-5
    H = np.fromiter(map(info, p), dtype=np.float).dot(p)
    if np.isnan(H) or np.isinf(H):
        print('Calculated inf/nan entropy')
        return 4.04
    return H

def entropy_report(x, y, bins):
    H = skinfo.entropy(x, bins=bins)
    kld = skinfo.kullback_leibler(x,y, bins=bins)
    
    return H, kld


def cluster_entropy(variograms, mean_shift, ax, ylabel=False, bins=None):
    c, H,intr, w = [], [], [], []
    data = np.asarray([v.experimental for v in variograms])
    pdata = pdist(data)
    if bins is None:
        bins = np.linspace(np.min(pdata), np.max(pdata), 25)
    
    for cl in np.unique(mean_shift.labels_):
        mem = data[np.where(mean_shift.labels_ == cl),][0]
        if len(mem) > cl_threshold:
            c.append(cl)
#            H.append(variogram_entropy(pdist(mem), bins))
            _h = skinfo.entropy(pdist(mem), bins)
            _kl = skinfo.kullback_leibler(pdist(mem), pdata, bins)
            intr.extend(len(mem) * [mean_shift.cluster_centers_[cl]])
            w.append(skinfo.entropy(pdist(mem), bins) * (len(mem) / len(data)))
            H.append(_h)
            print('H(#%d):   %.2f' % (cl, _h))
            print('KLD(#%d): %.3f' % (cl, _kl))
    intr = pdist(np.asarray(intr))
    #iloss = skinfo.kullback_leibler(intr, pdata, np.linspace(np.min(intr), np.max(intr), len(mean_shift.cluster_centers_)))
    iloss = skinfo.kullback_leibler(intr, pdata, np.linspace(np.min(pdata), np.max(pdata), 10))
    #print('Infomation loss: %.2f of %.2f' % (iloss, skinfo.entropy(pdata, bins)))
    print('Infomation loss: %.2f of %.2f' % (iloss, skinfo.entropy(pdata, np.linspace(np.min(pdata), np.max(pdata), 10))))
    print('Weighted Entropy: %.2f' % (np.sum(w)))
    ax.hlines(variogram_entropy(pdist(data), bins), 0, len(c) - 1, color='r', lw=3)
    ax.plot(range(len(c)), H, 'Dg', lw=2)
    ax.set_ylim(bottom=1)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(['#%d' % _ for _ in c])
    ax.set_xlabel('cluster')
    if ylabel:
        ax.set_ylabel('Entropy [bit]')
    return ax


def information_loss(variograms, mean_shift, bins=None, cl_threshold=10):
    _d = np.asarray([v.experimental for v in variograms])
    data = pdist(_d)
    if bins is None:
        bins = 25
    if isinstance(bins, int):
        bins = np.linspace(0, np.max(data), bins)
    intr = []
    for cl in np.unique(mean_shift.labels_):
        mem = data[np.where(mean_shift.labels_ == cl), ][0]
        if len(mem) > cl_threshold:
            intr.extend(len(mem) * [mean_shift.cluster_centers_[cl]])
    compressed = pdist(np.asarray(intr))
    
    return skinfo.kullback_leibler(compressed, data, bins), skinfo.entropy(data, bins)
