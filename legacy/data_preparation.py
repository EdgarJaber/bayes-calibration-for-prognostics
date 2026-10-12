import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def prepro_data_pce(data_hl, data_cl, n_max, n_samp, d):
    """
    Preprocess the THYC-Puffer-DEPOTHYC data for building a Gaussian Process metamodel.

    Parameters
    ----------
    data_hl : numpy array
        hot-leg data.
    data_cl : numpy array
        cold-leg data.
    simulation_time : numpy array
        Simulation time.

    n_max : int
        Maximum number of samples to consider.
    n_samp : int    
        Number of samples for the doe.
    d : int
        Dimension of the input parameters.

    Returns
    -------
    X : numpy array
        Input data.
    y : numpy array
        Output data.
    """

    output_sample = np.asarray([ (np.asarray(data_hl[:,1][i]) + np.asarray(data_cl[:,1][i]))/2 for i in range(n_max)])
    input_sample = np.asarray(data_hl[:,0])
    _input_sample = []

    for i in range(d):
        param = []
        for j in range(n_max):
            param.append(input_sample[:][j][i])
        _input_sample.append(param)
    _input_sample = np.asarray(_input_sample)

    ### OUTPUT SAMPLE PREPROCESSING ###
    delete_in = []
    for i in range(n_max):
        if np.isnan(output_sample[i]).any()==True :
            delete_in.append(i)

    print(len(delete_in), "nan trajectories in output sample")

    ### DELETE NAN VALUES ###
    X = np.delete(_input_sample, delete_in, 1)[:d-1,:n_samp].T
    y = np.delete(output_sample, delete_in, 0)[:n_samp,:]

    return X, y

def prepro_data_gp(data_hl, data_cl, simulation_time, n_max, n_samp, d, scale):
    """
    Preprocess the THYC-Puffer-DEPOTHYC data for building a Gaussian Process metamodel.

    Parameters
    ----------
    data_hl : numpy array
        hot-leg data.
    data_cl : numpy array
        cold-leg data.
    simulation_time : numpy array
        Simulation time.

    n_max : int
        Maximum number of samples to consider.
    n_samp : int    
        Number of samples for the doe.
    d : int
        Dimension of the input parameters.

    Returns
    -------
    X : numpy array
        Input data.
    y : numpy array
        Output data.
    Scaler : StandardScaler
        Scaler object.
    """

    output_sample = np.asarray([ (np.asarray(data_hl[:,1][i]) + np.asarray(data_cl[:,1][i]))/2 for i in range(n_max)])
    input_sample = np.asarray(data_hl[:,0])
    _input_sample = []

    for i in range(d):
        param = []
        for j in range(n_max):
            param.append(input_sample[:][j][i])
        _input_sample.append(param)
    _input_sample = np.asarray(_input_sample)

    ### OUTPUT SAMPLE PREPROCESSING ###
    delete_in = []
    for i in range(n_max):
        if np.isnan(output_sample[i]).any()==True :
            delete_in.append(i)

    ### DELETE NAN VALUES ###
    X = np.delete(_input_sample, delete_in, 1)[:,:n_samp].T
    y = np.delete(output_sample, delete_in, 0)[:n_samp,:]

    ### TIME AS INTEGER ###
    TIME = (simulation_time*24).astype(int)
    h_gv = np.linspace(0, 420481, 420481).astype(int)

    ### INTERPOLATION OF THE MISSING VALUES ###
    y_interpolated = np.asarray([np.interp(h_gv, TIME, y[i,:]) for i in range(len(y))])

    sample_times = X[:,d-1].astype(int)

    y = y_interpolated[np.where(sample_times), sample_times].T

    ### INPUT TIMES AS INTEGERS ###
    X[:,d-1] = X[:,d-1].astype(int)

    if scale == True:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler

    else:
        return X, y



def preprocess_data_gp(data_hl, data_cl, simulation_time, n_max, n_samp, d, scale=False):
    """
    Preprocess the THYC-Puffer-DEPOTHYC data for building a Gaussian Process metamodel.

    Parameters
    ----------
    data_hl : numpy array
        Hot-leg data with shape (n_samples, n_features).
    data_cl : numpy array
        Cold-leg data with shape (n_samples, n_features).
    simulation_time : numpy array
        Simulation time as an array.
    n_max : int
        Maximum number of samples to consider.
    n_samp : int
        Number of samples for the design of experiments (DoE).
    d : int
        Dimension of the input parameters.
    scale : bool, optional
        Whether to scale the input data using StandardScaler. Default is False.

    Returns
    -------
    X : numpy array
        Input data of shape (n_samp, d).
    y : numpy array
        Output data of shape (n_samp,).
    scaler : StandardScaler, optional
        The fitted scaler object, returned only if `scale=True`.
    """

    # Compute the output sample as the average of hot-leg and cold-leg values
    output_sample = np.mean([data_hl[:, 1], data_cl[:, 1]], axis=0)[:n_max]

    # Extract and reshape input sample
    input_sample = np.asarray(data_hl[:, 0][:n_max])
    input_sample_processed = np.array([input_sample[:, i] for i in range(d)])

    # Identify and remove NaN values in the output sample
    nan_indices = np.where(np.isnan(output_sample))[0]

    # Remove NaN entries from input and output samples
    X = np.delete(input_sample_processed, nan_indices, axis=1)[:, :n_samp].T
    y = np.delete(output_sample, nan_indices, axis=0)[:n_samp]

    # Convert simulation times to integers
    simulation_time_in_hours = (simulation_time * 24).astype(int)
    hourly_grid = np.arange(0, 420481).astype(int)

    # Interpolate missing values in the output data
    y_interpolated = np.array([
        np.interp(hourly_grid, simulation_time_in_hours, y_sample)
        for y_sample in y
    ])

    # Map input times to interpolated values
    sample_times = X[:, d-1].astype(int)
    y = np.array([y_interpolated[i, t] for i, t in enumerate(sample_times)])

    # Ensure input times are integers
    X[:, d-1] = sample_times

    # Optionally scale the input data
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler

    return X, y


def prepro_data_mlp(data_hl, data_cl, simulation_time, n_max, n_samp, d, scale):
    """
    Preprocess the THYC-Puffer-DEPOTHYC data for building a Gaussian Process metamodel.

    Parameters
    ----------
    data_hl : numpy array
        hot-leg data.
    data_cl : numpy array
        cold-leg data.
    simulation_time : numpy array
        Simulation time.

    n_max : int
        Maximum number of samples to consider.
    n_samp : int    
        Number of samples for the doe.
    d : int
        Dimension of the input parameters.

    Returns
    -------
    X : numpy array
        Input data.
    y : numpy array
        Output data.
    Scaler : StandardScaler
        Scaler object.
    """

    output_sample = np.asarray([ (np.asarray(data_hl[:,1][i]) + np.asarray(data_cl[:,1][i]))/2 for i in range(n_max)])
    input_sample = np.asarray(data_hl[:,0])
    _input_sample = []

    for i in range(d):
        param = []
        for j in range(n_max):
            param.append(input_sample[:][j][i])
        _input_sample.append(param)
    _input_sample = np.asarray(_input_sample)

    ### OUTPUT SAMPLE PREPROCESSING ###
    delete_in = []
    for i in range(n_max):
        if np.isnan(output_sample[i]).any()==True :
            delete_in.append(i)

    ### DELETE NAN VALUES ###
    X = np.delete(_input_sample, delete_in, 1)[:,:n_samp].T
    y = np.delete(output_sample, delete_in, 0)[:n_samp,:]

    ### TIME AS INTEGER ###
    TIME = (simulation_time*24).astype(int)
    h_gv = np.linspace(0, 420481, 420481).astype(int)

    ### INTERPOLATION OF THE MISSING VALUES ###
    y_interpolated = np.asarray([np.interp(h_gv, TIME, y[i,:]) for i in range(len(y))])

    sample_times = X[:,d-1].astype(int)

    y = y_interpolated[np.where(sample_times), sample_times].T

    ### INPUT TIMES AS INTEGERS ###
    X[:,d-1] = X[:,d-1].astype(int)

    if scale == True:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler

    else:
        return X, y


def prepro_data_kl(data_hl, data_cl, n_max, n_samp, d, scale=False):
    """
    Preprocess the THYC-Puffer-DEPOTHYC data for building a Karhunen-Loeve metamodel.

    Parameters
    ----------
    data_hl : numpy array
        hot-leg data.
    data_cl : numpy array
        cold-leg data.
    simulation_time : numpy array
        Simulation time.

    n_max : int
        Maximum number of samples to consider.
    n_samp : int    
        Number of samples for the doe.
    d : int
        Dimension of the input parameters.

    Returns
    -------
    X : numpy array
        Input data.
    y : numpy array
        Output data.
    """

    output_sample = np.asarray([ (np.asarray(data_hl[:,1][i]) + np.asarray(data_cl[:,1][i]))/2 for i in range(n_max)])
    input_sample = np.asarray(data_hl[:,0])
    _input_sample = []

    for i in range(d):
        param = []
        for j in range(n_max):
            param.append(input_sample[:][j][i])
        _input_sample.append(param)
    _input_sample = np.asarray(_input_sample)

    ### OUTPUT SAMPLE PREPROCESSING ###
    delete_in = []
    for i in range(n_max):
        if np.isnan(output_sample[i]).any()==True :
            delete_in.append(i)

    print(len(delete_in), "nan trajectories in output sample")

    ### DELETE NAN VALUES ###
    X = np.delete(_input_sample, delete_in, 1)[:d-1,:n_samp].T
    y = np.delete(output_sample, delete_in, 0)[:n_samp,:]

    if scale:
        scaler = StandardScaler()
        if d == 1:
            X = scaler.fit_transform(X.reshape(-1,1))
        else:
            X = scaler.fit_transform(X)
        return X, y, scaler
    
    else:
        return X, y
    
def preprocess_data_kl_1(data_hl, data_cl, n_max, n_samp, d, scale=False):
    """
    Preprocess the THYC-Puffer-DEPOTHYC data for building a Karhunen-Loève metamodel.

    Parameters
    ----------
    data_hl : numpy array
        Hot-leg data with shape (n_samples, n_features).
    data_cl : numpy array
        Cold-leg data with shape (n_samples, n_features).
    n_max : int
        Maximum number of samples to consider.
    n_samp : int
        Number of samples for the design of experiments (DoE).
    d : int
        Dimension of the input parameters.
    scale : bool, optional
        Whether to scale the input data using StandardScaler. Default is False.

    Returns
    -------
    X : numpy array
        Input data of shape (n_samp, d-1).
    y : numpy array
        Output data of shape (n_samp, n_features).
    scaler : StandardScaler, optional
        The fitted scaler object, returned only if `scale=True`.
    """

    # Compute the output sample as the average of hot-leg and cold-leg values
    output_sample = np.mean([data_hl[:, 1], data_cl[:, 1]], axis=0)[:n_max]

    # Extract input sample
    input_sample = data_hl[:, 0][:n_max]
    input_sample_processed = np.array([input_sample[:, i] for i in range(d)])

    # Identify and remove trajectories with NaN in the output sample
    nan_indices = np.where(np.isnan(output_sample))[0]
    print(f"{len(nan_indices)} NaN trajectories in output sample")

    # Remove NaN entries from input and output samples
    X = np.delete(input_sample_processed, nan_indices, axis=1)[:d-1, :n_samp].T
    y = np.delete(output_sample, nan_indices, axis=0)[:n_samp]

    # Optionally scale the input data
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y, scaler

    return X, y


def preprocess_field_data(csv_file: str, operation_time: np.array, cleaning_dates: list, with_time_division: bool):
    """
    Preprocess field data to make it compatible with the MCMC calibration procedure.
    
    Args:
        csv_file (str): Path to the CSV file containing field data.
        operation_time (np.array): Array of operation times.
        cleaning_dates (list): List of cleaning dates.
    
    Returns:
        tuple: A tuple containing preprocessed data and time indices.
    """
    esticol, etv = [], []
    esticol_times, etv_times = [], []

    # Read CSV file and extract data
    with open(csv_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for line in csv_reader:
            time_in_days = float(line['temps']) / 24
            if line['ETV'] in ['', '-']:
                esticol.append([time_in_days, float(line['ESTICOL'].strip('%'))])
                esticol_times.append(time_in_days)
            else:
                etv.append([time_in_days, float(line['ETV'].strip('%'))])
                etv_times.append(time_in_days)

    # Convert lists to numpy arrays
    esticol = np.asarray(esticol)
    etv = np.asarray(etv)
    esticol_times = np.asarray(esticol_times)
    etv_times = np.asarray(etv_times)


    if with_time_division:
        # Extend cleaning dates with start and end times
        cleaning_dates = [0] + cleaning_dates + [operation_time[-1]]

        # Split data by cleaning dates
        data_esticol = [
            esticol[(esticol_times >= cleaning_dates[i]) & (esticol_times < cleaning_dates[i + 1])]
            for i in range(len(cleaning_dates) - 1)
        ]
        data_etv = [
            etv[(etv_times >= cleaning_dates[i]) & (etv_times < cleaning_dates[i + 1])]
            for i in range(len(cleaning_dates) - 1)
        ]

        # Combine ESTICOL and ETV data
        data_combined = [
            np.sort(np.concatenate((data_esticol[i], data_etv[i]), axis=0), axis=0)
            for i in range(len(data_esticol))
        ]

        # Extract time indices for ESTICOL and ETV
        time_indices = [
            [
                np.where(np.isin(data_combined[i][:, 0], data_esticol[i][:, 0]))[0],
                np.where(np.isin(data_combined[i][:, 0], data_etv[i][:, 0]))[0]
            ]
            for i in range(len(data_combined))
        ]

        # Package data into final format
        data = [[data_esticol[i], data_etv[i]] for i in range(len(data_esticol))]

        return data, time_indices
    
    else:
        # Combine ESTICOL and ETV data
        data = np.sort(np.concatenate((esticol, etv), axis=0), axis=0)

        # Extract time indices for ESTICOL and ETV
        time_indices = [
            np.where(np.isin(data[:, 0], esticol[:, 0]))[0],
            np.where(np.isin(data[:, 0], etv[:, 0]))[0]
        ]

        return data, time_indices
