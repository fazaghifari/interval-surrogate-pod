import numpy as np
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.pod import POD
from src.ipod import IPOD
import chaospy
from src.isvd import tools
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

def load_data(PATH, n_train=15, reshape_size=None):
    """Load and extract data

    Args:
        PATH (str): path to dataset
        n_train (int, optional): number of training sample. Defaults to 15.

    Returns:
        _type_: _description_
    """
    # load and extract data
    data = scipy.io.loadmat(PATH)
    minx1 = data["minx1"].T
    maxx1 = data["maxx1"].T
    minx2 = data["minx2"].T
    maxx2 = data["maxx2"].T
    min_deflection = data["min_def"].T  # snapshot shape should be n_elements x n_sample
    max_deflection = data["max_def"].T  # snapshot shape should be n_elements x n_sample

    if reshape_size is not None:
        min_deflection = min_deflection.reshape(reshape_size[0],reshape_size[1])
        max_deflection = max_deflection.reshape(reshape_size[0],reshape_size[1])

    min_def_train = min_deflection[:,:n_train]
    max_def_train = max_deflection[:,:n_train]
    minx1_train = minx1[:n_train,:]
    maxx1_train = maxx1[:n_train,:]
    minx2_train = minx2[:n_train,:]
    maxx2_train = maxx2[:n_train,:]

    min_def_test = min_deflection[:,n_train:]
    max_def_test = max_deflection[:,n_train:]
    minx1_test = minx1[n_train:,:]
    maxx1_test = maxx1[n_train:,:]
    minx2_test = minx2[n_train:,:]
    maxx2_test = maxx2[n_train:,:]

    # Construct snapshot matrix
    # Interval snapshot shape should be n_elements x n_sample x 2
    intv_snapshot_train = np.dstack((min_def_train, max_def_train)) # times -1 because the value is negative
    intv_snapshot_test = np.dstack((min_def_test, max_def_test))

    # For bridge case, we would like to flip the sign
    if reshape_size is None:
        intv_snapshot_train = intv_snapshot_train * -1
        intv_snapshot_test = intv_snapshot_test * -1

    # construct params matrix
    # Interval snapshot shape should be n_sample x n_dimension x 2
    x_lower_train = np.concatenate([minx1_train,minx2_train], axis=1)
    x_upper_train = np.concatenate([maxx1_train,maxx2_train], axis=1)
    
    intv_x_train = np.dstack((x_lower_train, x_upper_train))

    # construct params matrix
    # Interval snapshot shape should be n_sample x n_dimension x 2
    x_lower_test = np.concatenate([minx1_test,minx2_test], axis=1)
    x_upper_test = np.concatenate([maxx1_test,maxx2_test], axis=1)
    intv_x_test = np.dstack((x_lower_test, x_upper_test))

    return (intv_x_train, intv_snapshot_train, intv_x_test, intv_snapshot_test)

def average_interval(int_mat):
    """return average of an interval matrix

    Args:
        int_mat (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 0.5*(int_mat[:,:,0] + int_mat[:,:,1])

def interval_pod(train_snapshot, rank=3, decomp_strat = "b"):
    """Perform interval proper orthogonal decomposition

    Args:
        train_snapshot (np.ndarray): _description_
        rank (interval POD target rank): _description_

    Returns:
        _type_: _description_
    """
    ipod = IPOD(target_rank = rank, decomp_strategy= decomp_strat)
    ipod.fit(snapshot= train_snapshot)

    return ipod

def standardize(x, x_domain):
    """
    Standardize an array to 0-1 given the input array and the minimum and maximum values of the domain.
    """
    return (x - x_domain[0]) / (x_domain[1] - x_domain[0]) * 2 - 1 

def train_pce(x, pod_coeff, x1_domain, x2_domain, fit_range=False):
    """Train PCE model

    Args:
        x (_type_): _description_
        pod_coeff (_type_): _description_
        x1_domain (_type_): _description_
        x2_domain (_type_): _description_

    Returns:
        _type_: _description_
    """
    # x shape should be n_sample x n_dim
    # Train multiple PCE models
    x1 = chaospy.Uniform(0,1)
    x2 = chaospy.Uniform(0,1)
    joint = chaospy.J(x1,x2)
    expansion = chaospy.generate_expansion(2, joint)
    pce_list = []

    if not fit_range:  # If mean regression only
        model = LinearRegression(fit_intercept=False)

        x_norm = x.copy()
        x_norm[:,0] = standardize(x_norm[:,0], x1_domain)
        x_norm[:,1] = standardize(x_norm[:,1], x2_domain)

        w_train = np.ones(x.shape[0]) / x.shape[0]

        for i in range(pod_coeff.shape[1]):
            surrogate,coefs = chaospy.fit_regression(expansion, x_norm.T, pod_coeff[:,i].reshape(-1,1), 
                                                    model=model, retall=True)
            pce_list.append(surrogate)

    else:  # If 
        # Break pod_coeff into 2 components:
        pod_coeff_mean = average_interval(pod_coeff)
        pod_coeff_radii = (pod_coeff[:,:,1] - pod_coeff[:,:,0])/2

        # Interval mean regression
        model_mean = LinearRegression(fit_intercept=False)

        x_mean = x[:,:,0].copy()
        x_mean[:,0] = standardize(x_mean[:,0], x1_domain)
        x_mean[:,1] = standardize(x_mean[:,1], x2_domain)

        w_train = np.ones(x.shape[0]) / x.shape[0]
        pce_means = []
        for i in range(pod_coeff_mean.T.shape[1]):
            surrogate,coefs = chaospy.fit_regression(expansion, x_mean.T, pod_coeff_mean.T[:,i].reshape(-1,1), 
                                                    model=model_mean, retall=True)
            pce_means.append(surrogate)
        
        # Interval radii reg
        model_radii = LinearRegression(fit_intercept=False, positive=True)

        x_radii = x[:,:,1].copy()
        pce_radii = []
        for i in range(pod_coeff_radii.T.shape[1]):
            surrogate,coefs = chaospy.fit_regression(expansion, x_mean.T, pod_coeff_radii.T[:,i].reshape(-1,1), 
                                                    model=model_radii, retall=True)
            pce_radii.append(surrogate)

        pce_list = [pce_means, pce_radii]
    
    return pce_list

def predict(x_in, pce_list, latent_vec, x1_domain, x2_domain, pred_range=False):
    """predict snapshot given arbitrary parameters and list of pce models

    Args:
        x_in (np.ndarray): Arbitrary physical parameters
        pce_list (list): list of PCE models
        latent_vec (np.ndarray): interval POD latent vectors

    Returns:
        _type_: _description_
    """
    if not pred_range:
        x_norm = x_in.copy()
        x_norm[:,0] = standardize(x_norm[:,0], x1_domain)
        x_norm[:,1] = standardize(x_norm[:,1], x2_domain)

        pred_coeff=[]
        single_param = np.array([pce(*x_norm.T).T[:,0] for pce in pce_list])
        pred_coeff.append(single_param)

        pred_coeff = np.concatenate(pred_coeff, axis=0)

        predicted_snapshot = tools.interval_matmul(latent_vec, np.dstack((pred_coeff, pred_coeff)))
    
    else:
        # breakdown pce list
        pce_mean = pce_list[0]
        pce_radii = pce_list[1]

        # if pred_range, x_in has 3 dimensions (n x m x 2) with the 1st array on the 3rd axis as the x_mean
        #  and the 2nd array as the x_radii
        # predict mean
        x_mean = x_in[:,:,0].copy()
        x_mean[:,0] = standardize(x_mean[:,0], x1_domain)
        x_mean[:,1] = standardize(x_mean[:,1], x2_domain)

        pred_mean=[]
        single_param = np.array([pce(*x_mean.T).T[:,0] for pce in pce_mean])
        pred_mean.append(single_param)
        pred_mean = np.concatenate(pred_mean, axis=0)

        # predict radii
        x_radii = x_in[:,:,1].copy()

        pred_radii=[]
        single_radii = np.array([pce(*x_radii.T).T[:,0] for pce in pce_radii])
        pred_radii.append(single_radii)
        pred_radii = np.concatenate(pred_radii, axis=0)

        pred_coef1 = pred_mean + pred_radii
        pred_coef2 = pred_mean - pred_radii

        pred_coeff = np.dstack((pred_coef1, pred_coef2))
        predicted_snapshot = tools.interval_matmul(np.dstack((latent_vec,latent_vec)), pred_coeff)



    return predicted_snapshot

def RMSE(a,b):
    return np.sqrt(np.mean((a-b)**2))

def field_plot(reference, predicted, sim_id=42, reshape_size=(100,100), 
               filename3d="images/tempprofile3d.png", cross_sec=None):
    """_summary_

    Args:
        reference (np.array): numpy array of the reference snapshot matrix
        predicted (np.array): numpy array of the predicted snapshot matrix
        sim_id (int): random number to select one sample from the data
        reshape_size (tuple): size of the field grid
        filename3d (str): filename string for the 3d plot
        cross_section (str): filename string for the cross-section plot, default to None.
    """
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    mpl.rc('text', usetex=True)
    mpl.rcParams.update({'font.size': 14.5, 'font.weight':'bold'})
    plt.rcParams['text.latex.preamble'] = r'\boldmath'

    lower_ref = reference[:,sim_id,0]
    upper_ref = reference[:,sim_id,1]
    lower_pred = predicted[:,sim_id,0]
    upper_pred = predicted[:,sim_id,1]

    upper_temp_ref = upper_ref.reshape(reshape_size)
    lower_temp_ref = lower_ref.reshape(reshape_size)
    upper_temp_pred = upper_pred.reshape(reshape_size)
    lower_temp_pred = lower_pred.reshape(reshape_size)

    # Plot 3D profile
    ax_ranges = [-0.5, 0.5, -0.5, 0.5, 0, 6]
    ax_scale = [1.0, 1.0, 1.0]
    ax_extent = ax_ranges * np.repeat(ax_scale, 2)
    x = np.linspace(-0.5, 0.5, reshape_size[0])
    y = np.linspace(-0.5, 0.5, reshape_size[0])
    mx, my = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    norm = mpl.colors.Normalize(vmin=np.min(upper_temp_pred), vmax=np.max(upper_temp_pred))
    cmap = mpl.cm.jet

    surf_pred = ax.plot_surface(mx, my, upper_temp_pred, alpha = 1, rstride=1, 
                                cstride=1, facecolors=cmap(norm(upper_temp_pred)), linewidth=0.5, antialiased=True)
    surf_ref = ax.plot_surface(mx, my, upper_temp_ref, alpha = 1, rstride=1, 
                               cstride=1, facecolors=cmap(norm(upper_temp_ref)),linewidth=0.5, antialiased=True)
    ax.view_init(elev=14., azim=140)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("T ($^\circ C$)")
    plt.savefig(filename3d, dpi=400, format="png")
    plt.show()

    if cross_sec is not None:
        pred_profile_up = upper_temp_pred[:,75]
        ref_profile_up = upper_temp_ref[:,75]
        pred_profile_lo = lower_temp_pred[:,75]
        ref_profile_lo = lower_temp_ref[:,75]
        plt.plot(np.linspace(-0.5, 0.5, reshape_size[0]), pred_profile_up, "r--", label=r"\textbf{upperbound CCRM}")
        plt.plot(np.linspace(-0.5, 0.5, reshape_size[0]), ref_profile_up, "^", label=r"\textbf{upperbound actual}")
        plt.plot(np.linspace(-0.5, 0.5, reshape_size[0]), pred_profile_lo, "b-.", label=r"\textbf{lowerbound CCRM}")
        plt.plot(np.linspace(-0.5, 0.5, reshape_size[0]), ref_profile_lo, "s", label=r"\textbf{lowerbound actual}")
        plt.legend()
        plt.xlabel("$\mathbf{x}$", weight="bold")
        plt.ylabel("$\mathbf{T} (\mathbf{^\circ C})$", weight="bold")
        plt.savefig(cross_sec, dpi=400, format="png")
        plt.show()


def error_plot(meanreg_data, ccrm_data, data_fraction, savefile="images/bridge_rmse.png", ylim=[0.0001, 0.0007]):
    """Function for plotting error

    Args:
        meanreg_data (list): list of mean regression data.
        ccrm_data (list): list of ccrm regression data.
        data_fraction (list): list of data fraction that evaluated.
        savefile (str): image save file location
        ylim (list): y axis limits
    """

    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    mpl.rc('text', usetex=True)
    mpl.rcParams.update({'font.size': 14.5, 'font.weight':'bold'})
    plt.rcParams['text.latex.preamble'] = r'\boldmath'

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(14,6))
    ax1.plot(data_fraction, meanreg_data[0], "r-^", label=r"\textbf{lowerbound rmse homogenous}")
    ax1.plot(data_fraction, meanreg_data[1], "b-o", label=r"\textbf{upperbound rmse homogenous}")
    ax1.plot(data_fraction, meanreg_data[2], "r--", label=r"\textbf{lowerbound rmse heterogenous}")
    ax1.plot(data_fraction, meanreg_data[3], "b-.", label=r"\textbf{upperbound rmse heterogenous}")
    ax1.set_title(r"\textbf{Mean Regression}")

    ax2.plot(data_fraction, ccrm_data[0], "r-^", label=r"\textbf{lowerbound rmse homogenous}")
    ax2.plot(data_fraction, ccrm_data[1], "b-o", label=r"\textbf{upperbound rmse homogenous}")
    ax2.plot(data_fraction, ccrm_data[2], "r--", label=r"\textbf{lowerbound rmse heterogenous}")
    ax2.plot(data_fraction, ccrm_data[3], "b-.", label=r"\textbf{upperbound rmse heterogenous}")
    ax2.set_title(r"\textbf{CCRM Regression}")

    ax1.set_xlabel(r"$\%$ \textbf{of training data fraction}")
    ax2.set_xlabel(r"$\%$ \textbf{of training data fraction}")
    ax1.set_ylabel(r"\textbf{RMSE} ($m$)")

    ax1.set_ylim(ylim)
    ax1.set_xticks(data_fraction)
    ax2.set_xticks(data_fraction)

    plt.legend()
    plt.savefig(savefile, dpi=400, format="png")
    plt.show()

def violation_check(actual_snap, pred_snap):
    # lower bound
    # predicted lowerbound <= actual lowerbound
    actual_lb = actual_snap[:,:,0]
    pred_lb = pred_snap[:,:,0]
    n_violate_lb = 0
    for col in range(actual_lb.shape[1]):
        violation_lb = np.sum(pred_lb[:,col] > actual_lb[:,col])
        n_violate_lb += np.minimum(1,violation_lb)  # if violation > 1, assign 1. else, 0
    lb_violation = n_violate_lb/actual_lb.shape[1]

    # upper bound
    # predicted upperbound >= actual upperbound
    actual_ub = actual_snap[:,:,1]
    pred_ub = pred_snap[:,:,1]
    n_violate_ub = 0
    for col in range(actual_ub.shape[1]):
        violation_ub = np.sum(pred_ub[:,col] < actual_ub[:,col])
        n_violate_ub += np.minimum(1,violation_ub)  # if violation > 1, assign 1. else, 0
    ub_violation = n_violate_ub/actual_ub.shape[1]

    return ub_violation, lb_violation

def main(config):
    """Main function
    """

    path = config["PATH"]
    n_train = config["N_TRAIN"]
    pod_rank = config["POD_RANK"]
    x1_domain = config["X1_DOMAIN"]
    x2_domain = config["X2_DOMAIN"]
    fit_range = config["FIT_RANGE"]  # Boolean: Switch to turn CCRM
    reshape_size = config["RESHAPE_SIZE"]

    # load data
    intv_x_train, intv_snapshot_train, intv_x_test, intv_snapshot_test = load_data(path, n_train, reshape_size)
    intv_x_mean_train = average_interval(intv_x_train)
    intv_x_mean_test = average_interval(intv_x_test)
    x_radii_train = (intv_x_train[:,:,1] - intv_x_train[:,:,0])/2
    x_radii_test = (intv_x_test[:,:,1] - intv_x_test[:,:,0])/2

    radii_scaler = MinMaxScaler()
    x_radii_train_scaled = radii_scaler.fit_transform(x_radii_train)
    x_radii_test_scaled = radii_scaler.fit_transform(x_radii_test)

    if not fit_range:
        decomp_strat = "b"
        xtrain = intv_x_mean_train
        xtest = intv_x_mean_test
    else:
        decomp_strat = "b_intv"
        xtrain = np.dstack((intv_x_mean_train, x_radii_train_scaled))
        xtest = np.dstack((intv_x_mean_test, x_radii_test_scaled))

    # Construct the interval POD
    ipod = interval_pod(intv_snapshot_train, pod_rank, decomp_strat=decomp_strat)
    latent_vec = ipod.latent_vec
    pod_coeff = ipod.pod_coeff

    # train PCE model
    pce_list = train_pce(xtrain, pod_coeff, x1_domain, x2_domain, fit_range=fit_range)

    # predict test
    predict_snap = predict(xtest, pce_list, latent_vec, x1_domain, x2_domain, pred_range=fit_range)

    # RMSE
    rmse_lobound = RMSE(intv_snapshot_test[:,:,0], predict_snap[:,:,0])
    rmse_upbound = RMSE(intv_snapshot_test[:,:,1], predict_snap[:,:,1])

    return rmse_lobound, rmse_upbound, intv_snapshot_test, predict_snap, pod_coeff