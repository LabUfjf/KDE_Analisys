def ash(data,m=10,tip='NEAREST'):

    """
    #ASH  AVERAGE SHIFT HISTOGRAM.
     
    [X,Y] = ASH(DATA) bins the elements of Y into FD equally spaced
    containers with 10 SHIFT HISTOGRAMS and returns the mean number
    of elements in each container.
    
    [X,Y] = ASH(DATA,M), where M is a scalar, uses M SHIFT HISTOGRAMS.
    
    [X,Y] = ASH(DATA,X,TYPE), where TYPE is a defined, the interpolation
    has made with the definition. OBS. the default value is 'NEAREST'.
    
    ASH(...) without output arguments produces a histogram bar plot of
    the results.
    
    Class support for inputs Y, X: 
       float: double, single
    
     Parse possible Axes input
    
    """

import numpy as np

binning = int(np.round(bin_fd(data)))

if binning < 10 :
    binning=10

h=(np.max(data)-np.min(data))/binning;

%% AVERAGE SHIFT HISTOGRAM


t0=np.linspace(0,(h-(h/m)),m)

fa = xa = np.zeros((m,binning-1),dtype=float)


for j in range(m):
    gridx=np.arange((np.min(data)+t0[j]),(np.max(data)+t0[j]),h)
    for i in range(binning-1):
        fa[j,i]=np.size(np.where(np.logical_and(data>=gridx[i], data<=gridx[i+1])))
        xa[j,i]=(gridx[i]+gridx[i+1])/2;

newgrid = np.reshape(xa,(1,np.size(xa)))

##PAREI AQUI

for k in range(m-1):

    fsh(k,:)=interp1(xa(k,:),fa(k,:),newgrid,type,0);


fash=mean(fsh,1);

if nargout > 0
    x=newgrid;
    y=fash;
else
    bar(newgrid,fash,1,'hist')
end

def bin_fd(x):
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 1 for the number of bins.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)