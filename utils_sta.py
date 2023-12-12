import numpy as np
import os
import itertools 
from functools import reduce
from functools import wraps
import matplotlib.pyplot as plt
from matplotlib import animation, cm, gridspec
import scipy.signal
from matplotlib.patches import Ellipse

#Code adapted from Baccus lab


def plotwrapper(func):
    """Decorator that adds axis and figure keyword arguments to the kwargs
    of a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        figsize = kwargs.get('figsize', None)

        if 'ax' not in kwargs:
            if 'fig' not in kwargs:
                kwargs['fig'] = plt.figure(figsize=figsize)
            kwargs['ax'] = kwargs['fig'].add_subplot(111)
        else:
            if 'fig' not in kwargs:
                kwargs['fig'] = kwargs['ax'].get_figure()

        func(*args, **kwargs)
        # plt.show()
        plt.draw()
        return kwargs['fig'], kwargs['ax']

    return wrapper

def compute_STA_splitfiles(all_spikes,stim_path,nbins_before,STA_files,nbins_after = 0,to_norm = True):
	nb,na = nbins_before, nbins_after
	ste_it_lst = []
	for file in STA_files:
		stimulus = np.load(os.path.join(stim_path,str(file+1)+'.npy'))
		if to_norm:
			stimulus = stimulus/255 #Value from (0 to 1)
		spikes = all_spikes[file,:]

		ste_it = ((spikes[idx]*stimulus[(idx -nb):(idx + na),:,:].astype('float64')) 
					for idx in np.where(spikes > 0)[0]
					if (idx > nb and (idx + na) < stimulus.shape[0]))

		ste_it_lst.append(ste_it)

	ste_it = itertools.chain(*ste_it_lst)
	try:
		first = next(ste_it)
		sta = reduce(lambda sta, x: np.add(sta,x),ste_it,first) / float(sum(spikes))
	except StopIteration as e:
		print('No iterators')
		return None

	return sta

def decompose(sta):
	_,u,_,v = lowranksta(sta,k=1)
	return v[0].reshape(sta.shape[1:]),u[:,0]

def lowranksta(sta_orig,k=10):
	f = sta_orig.copy() - sta_orig.mean()

	assert f.ndim >= 2, "STA must be at least 2-D"
	u, s, v = np.linalg.svd(flat2d(f), full_matrices=False)

	k = np.min([k, s.size])
	u = u[:, :k]
	s = s[:k]
	v = v[:k, :]

	sk = (u.dot(np.diag(s).dot(v))).reshape(f.shape)

	sign = np.sign(np.tensordot(u[:, 0], f, axes=1).sum())
	u *= sign
	v *= sign

	return sk, u, s, v
def flat2d(x):
	return x.reshape(x.shape[0], -1)

def get_ellipse(spatial_filter,sigma = 2.):
	# preprocess
	zdata = normalize_spatial(spatial_filter, clip_negative=True)
	zdata /= zdata.max()

	# get initial parameters
	nx, ny = spatial_filter.shape
	xm, ym = np.meshgrid(np.arange(nx), np.arange(ny))
	pinit = _initial_gaussian_params(xm, ym, zdata)

	# optimize
	data = np.vstack((xm.ravel(), ym.ravel()))
	popt, pcov = scipy.optimize.curve_fit(_gaussian_function,
										  data,
										  zdata.ravel(),
										  p0=pinit,
										  maxfev=10000)

	# return ellipse parameters, scaled by the appropriate scale factor
	return _popt_to_ellipse(*popt, sigma=sigma)

def normalize_spatial(frame, scale_factor=1.0, clip_negative=False):
	f = frame.copy()
	f = f - f.mean()

	# compute the mean of pixels within +/- 5 std. deviations of the mean
	outlier_threshold = 5 * np.std(f.ravel())
	mu = f[(f <= outlier_threshold) & (f >= -outlier_threshold)].mean()

	# normalize by the standard deviation of the pixel values
	f_centered = f - mu
	f_centered = f_centered/f_centered.std()

	# resample by the given amount
	f_resampled = resample(f_centered, scale_factor)

	# clip negative values
	if clip_negative:
		f_resampled = np.maximum(f_resampled, 0)

	return f_resampled
def resample(arr, scale_factor):
	assert scale_factor > 0, "Scale factor must be non-negative"

	if arr.ndim == 1:
		return scipy.signal.resample(arr,
				int(np.ceil(scale_factor * arr.size)))

	elif arr.ndim == 2:
		assert arr.shape[0] == arr.shape[1], "Array must be square"
		n = int(np.ceil(scale_factor * arr.shape[0]))
		return scipy.signal.resample(
				scipy.signal.resample(arr, n, axis=0), n, axis=1)

	else:
		raise ValueError('Input array must be either 1-D or 2-D')

def _initial_gaussian_params(xm, ym, z, width=5):
	xi = z.sum(axis=0).argmax()
	yi = z.sum(axis=1).argmax()
	yc = xm[xi, yi]
	xc = ym[xi, yi]

	# compute precision matrix entries
	a = 1 / width
	b = 0
	c = 1 / width

	return xc, yc, a, b, c

def _popt_to_ellipse(y0, x0, a, b, c, sigma=2.):
	u, v = np.linalg.eigh(np.array([[a, b], [b, c]]))

	# convert precision standard deviations
	scale = sigma * np.sqrt(scipy.stats.chi2.ppf(0.6827, df=2))
	scaled_sigmas = scale * np.sqrt(1 / u)

	# rotation angle
	theta = np.rad2deg(np.arccos(v[1, 1]))

	return (x0, y0), scaled_sigmas, theta

# @plotwrapper
def spatial(filt, dx=1.0, maxval=None, color='seismic', **kwargs):
	"""
	Plot the spatial component of a full linear filter.
	If the given filter is 2D, it is assumed to be a 1D spatial filter,
	and is plotted directly. If the filter is 3D, it is decomposed into
	its spatial and temporal components, and the spatial component is plotted.
	Parameters
	----------
	filt : array_like
		The filter whose spatial component is to be plotted. It may have
		temporal components.
	dx : float, optional
		The spatial sampling rate of the STA, setting the scale of the
		x- and y-axes.
	maxval : float, optional
		The value to use as minimal and maximal values when normalizing the
		colormap for this plot. See ``plt.imshow()`` documentation for more
		details.
	ax : matplotlib Axes object, optional
		The axes on which to plot the data; defaults to creating a new figure.
	Returns
	-------
	fig : matplotlib.figure.Figure
		The figure onto which the spatial STA is plotted.
	ax : matplotlib Axes object
		Axes into which the spatial STA is plotted.
	"""
	_ = kwargs.pop('fig')
	ax = kwargs.pop('ax')

	if filt.ndim > 2:
		spatial_filter, _ = decompose(filt)
	else:
		spatial_filter = filt.copy()

	# adjust color limits if necessary
	if not maxval:
		spatial_filter = spatial_filter - np.mean(spatial_filter)
		maxval = np.max(np.abs(spatial_filter))

	# plot the spatial component
	spatial_range = (0.0, spatial_filter.shape[0] * dx, 
					 0.0, spatial_filter.shape[1] * dx)
	ax.imshow(spatial_filter,
			  cmap=color,#gray,
			  interpolation='nearest',
			  aspect='equal',
			  vmin=-maxval,
			  vmax=maxval,
			  extent=spatial_range,
			  **kwargs)
    # return spatial_filter



def plot_sta(time, sta, dx=1.0):
	"""
	Plot a linear filter.
	If the given filter is 1D, it is direclty plotted. If it is 2D, it is
	shown as an image, with space and time as its axes. If the filter is 3D,
	it is decomposed into its spatial and temporal components, each of which
	is plotted on its own axis.
	Parameters
	----------
	time : array_like
		A time vector to plot against.
	dx : float, optional
		The spatial sampling rate of the STA, setting the scale of the
		x- and y-axes.
	sta : array_like
		The filter to plot.
	Returns
	-------
	fig : matplotlib.figure.Figure
		The figure onto which the STA is plotted.
	ax : matplotlib Axes object
		Axes into which the STA is plotted
	"""

	# plot 1D temporal filter
	if sta.ndim == 1:
		fig = plt.figure()
		fig, ax = temporal(time, sta, ax=fig.add_subplot(111))

	# plot 2D spatiotemporal filter
	elif sta.ndim == 2:

		# normalize
		stan = (sta - np.mean(sta)) / np.var(sta)

		# create new axes
		fig = plt.figure()
		fig, ax = spatial(stan, dx=dx, ax=fig.add_subplot(111))

	# plot 3D spatiotemporal filter
	elif sta.ndim == 3:

		# build the figure
		fig = plt.figure()
		gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

		# decompose
		spatial_profile, temporal_filter = ft.decompose(sta)

		# plot spatial profile
		_, axspatial = spatial(spatial_profile, dx=dx,
				ax=fig.add_subplot(gs[0]))

		# plot temporal profile
		fig, axtemporal = temporal(time, temporal_filter,
				ax=fig.add_subplot(gs[1]))
		axtemporal.set_xlim(time[0], time[-1])
		axtemporal.spines['right'].set_color('none')
		axtemporal.spines['top'].set_color('none')
		axtemporal.yaxis.set_ticks_position('left')
		axtemporal.xaxis.set_ticks_position('bottom')

		# return handles
		ax = (axspatial, axtemporal)

	else:
		raise ValueError('The sta parameter has an invalid '
				'number of dimensions (must be 1-3)')

	return fig, ax

@plotwrapper
def ellipse(filt, sigma=2.0, alpha=0.8, fc='none', ec='black', 
		lw=3, dx=1.0, **kwargs):
	"""
	Plot an ellipse fitted to the given receptive field.
	Parameters
	----------
	filt : array_like
		A linear filter whose spatial extent is to be plotted. If this
		is 2D, it is assumed to be the spatial component of the receptive
		field. If it is 3D, it is assumed to be a full spatiotemporal
		receptive field; the spatial component is extracted and plotted.
	sigma : float, optional
		Determines the threshold of the ellipse contours. This is
		the standard deviation of a Gaussian fitted to the filter 
		at which the contours are plotted. Default is 2.0.
	alpha : float, optional
		The alpha blending value, between 0 (transparent) and
		1 (opaque) (Default: 0.8).
	fc : string, optional
		Ellipse face color. (Default: none)
	ec : string, optional
		Ellipse edge color. (Default: black)
	lw : int, optional
		Line width. (Default: 3)
	dx : float, optional
		The spatial sampling rate of the STA, setting the scale of the
		x- and y-axes.
	ax : matplotlib Axes object, optional
		The axes onto which the ellipse should be plotted.
		Defaults to a new figure.
	Returns
	-------
	fig : matplotlib.figure.Figure
		The figure onto which the ellipse is plotted.
	ax : matplotlib.axes.Axes
		The axes onto which the ellipse is plotted.
	"""
	_ = kwargs.pop('fig')
	ax = kwargs.pop('ax')

	if filt.ndim == 2:
		spatial_filter = filt.copy()
	elif filt.ndim == 3:
		spatial_filter = decompose(filt)[0]
	else:
		raise ValueError('Linear filter must be 2- or 3-D')

	# get the ellipse parameters
	center, widths, theta = get_ellipse(spatial_filter, sigma=sigma)

	# compute parameters given spatial scale
	center, widths = map(lambda x: np.asarray(x) * dx, (center, widths))

	# create the ellipse
	center = (center[1],center[0])
	ell = Ellipse(xy=center, width=widths[0], height=widths[1], angle=theta,
				  alpha=alpha, ec=ec, fc=fc, lw=lw, **kwargs)

	ax.add_artist(ell)
	ax.set_xlim(0, spatial_filter.shape[0] * dx)
	ax.set_ylim(0, spatial_filter.shape[1] * dx)

@plotwrapper
def plot_cells(cells, dx=1.0, **kwargs):
	"""
	Plot the spatial receptive fields for multiple cells.
	Parameters
	----------
	cells : list of array_like
		A list of spatiotemporal receptive fields, each of which is
		a spatiotemporal array.
	dx : float, optional
		The spatial sampling rate of the STA, setting the scale of the
		x- and y-axes.
	ax : matplotlib Axes object, optional
		The axes onto which the ellipse should be plotted.
		Defaults to a new figure.
	Returns
	------
	fig : matplotlib.figure.Figure
		The figure onto which the ellipses are plotted.
	ax : matplotlib.axes.Axes
		The axes onto which the ellipses are plotted.
	"""
	_ = kwargs.pop('fig')
	ax = kwargs.pop('ax')
	colors = cm.Set1(np.random.rand(len(cells),))

	# for each cell
	for color, sta in zip(colors, cells):

		# get the spatial profile
		try:
			spatial_profile = decompose(sta)[0]
		except np.linalg.LinAlgError:
			continue

		# plot ellipse
		try:
			ellipse(spatial_profile, fc=color, ec=color,
					lw=2, dx=dx, alpha=0.3, ax=ax)
		except RuntimeError:
			pass

# def _initial_gaussian_params(xm, ym, z, width=5):
#     """
#     Guesses the initial 2D Gaussian parameters given a spatial filter.
#     Parameters
#     ----------
#     xm : array_like
#         The x-points for the filter.
#     ym : array_like
#         The y-points for the filter.
#     z : array_like
#         The actual data the parameters of which are guessed.
#     width : float, optional
#         The expected 1 s.d. width of the RF, in samples. (Default: 5)
#     Returns
#     -------
#     xc, yc : float
#         Estimated center points for the data.
#     a, b, c : float
#         Upper-left, lower-right, and off-diagonal terms for the estimated
#         precision matrix.
#     """

#     # estimate means
#     xi = z.sum(axis=0).argmax()
#     yi = z.sum(axis=1).argmax()
#     yc = xm[xi, yi]
#     xc = ym[xi, yi]

#     # compute precision matrix entries
#     a = 1 / width
#     b = 0
#     c = 1 / width

#     return xc, yc, a, b, c
def _gaussian_function(data, x0, y0, a, b, c):
    """
    A 2D gaussian function (used for fitting an ellipse to RFs)
    Parameters
    ----------
    data : array_like
        A (2 by N) array of N data points
    x0 : float
        The x-location of the center of the ellipse
    y0 : float
        The y-location of the center of the ellipse
    a : float
        The upper left number in the precision matrix
    b : float
        The upper right / lower left number in the precision matrix
    c : float
        The lower right number in the precision matrix
    Returns
    -------
    z : array_like
        The (unnormalized) values of the 2D gaussian function with the given
        parameters
    """

    # center the data
    xc = data[0] - x0
    yc = data[1] - y0

    # gaussian function
    return np.exp(-0.5 * (a * xc**2 + 2 * b * xc * yc + c * yc**2))
