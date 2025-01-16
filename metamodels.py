import numpy as np
import openturns as ot

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import DataLoader, TensorDataset


class GpMetamodel(object):
    """
    Wrapper for OpenTURNS Gaussian Process.
    """
    def __init__(
        self, trend: str, kernel: str, 
        dimension:int, noise: float = None
    ) -> None:
        self.trend = trend
        self.kernel = kernel
        self.dimension = dimension
        self.trained_ = False
        self.noise = noise

    def fit(self, X_train, y_train):

        if self.trend not in ['Constant', 'Linear', 'Quad']:
            raise ValueError(f"trend must be one of ['Constant', 'Linear', 'Quad']")

        if self.kernel not in ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']:
            raise ValueError(f"kernel must be one of ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']")
        
        if self.trend == 'Constant':
            basis = ot.ConstantBasisFactory(self.dimension).build()
        elif self.trend == 'Linear':
            basis = ot.LinearBasisFactory(self.dimension).build()
        elif self.trend == 'Quad':
            basis = ot.QuadraticBasisFactory(self.dimension).build()    

        if self.kernel == 'AbsExp':
            covarianceModel = ot.AbsoluteExponential([1.0]*self.dimension)
        elif self.kernel == 'SqExp':
            covarianceModel = ot.SquaredExponential([1.0]*self.dimension)
        elif self.kernel == 'M-1/2':
            covarianceModel = ot.MaternModel([1.0]*self.dimension, [1.0], 0.5)
        elif self.kernel == 'M-3/2':
            covarianceModel = ot.MaternModel([1.0]*self.dimension, [1.0], 1.5)
        elif self.kernel == 'M-5/2':
            covarianceModel = ot.MaternModel([1.0]*self.dimension, [1.0], 2.5)

        if self.noise:
            covarianceModel.setNuggetFactor(self.noise)

        self.gp = ot.KrigingAlgorithm(
            ot.Sample(X_train),
            ot.Sample(y_train.reshape(-1, 1)),
            covarianceModel, basis
        )

        self.gp.run()

        self.trained_ = True

    def predict(self, X_test, return_std=False):

        metamodel = self.gp.getResult()(X_test)

        y_pred = metamodel.getMean()
        y_std = metamodel.getStandardDeviation()

        if not return_std:
            return np.array(y_pred)
        else:
            return np.array(y_pred), np.array(y_std)

    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False



class VPCEMetamodel(object):
    """
    Vector Polynomial chaos expansions with regression method for coefficient estimation.
    Comes with a fit and predict method.
    """
    def __init__(
            self, degree=5, 
            q_norm=2.0,
            verbose=True
            )-> None:
        self.degree = degree
        self.q_norm = q_norm
        self.verbose = verbose
        self.input_dimension = 1
        self.trained_ = False

    def fit(self, X_train, y_train, prior_distribution_list, distribution):

        self.prior_distribution_list = prior_distribution_list
        self.distribution = distribution
        
        multivar_basis = ot.OrthogonalProductPolynomialFactory(self.marginals)
        selection_algo = ot.LeastSquaresMetaModelSelectionFactory()
        projection_strategy = ot.LeastSquaresStrategy(X_train, y_train, selection_algo)
        enum_func = ot.HyperbolicAnisotropicEnumerateFunction(self.input_dimension, self.q_norm)
        P = enum_func.getBasisSizeFromTotalDegree(self.degree)
        adaptive_strategy = ot.FixedStrategy(multivar_basis, P)
        self.chaos_algo = ot.FunctionalChaosAlgorithm(X_train, y_train, distribution, adaptive_strategy, projection_strategy)
        self.chaos_algo.run()
        if self.verbose:
            print(f"Running Polynomial Chaos Expansion with regression method for degree {self.degree}, q-norm {self.q_norm}")
        self.pce = self.chaos_algo.getResult()
        self.trained_ = True

    def predict(self, X_test):
        if not self.trained_:
            raise ValueError("You must first fit the Polynomial Chaos Expansion")
        X_test = StandardScaler().fit_transform(X_test)
        y_pred = self.pce(X_test)
        return y_pred
    
    def r2_score(self, X_test, y_test):
        if not self.trained_:
            raise ValueError("You must first fit the Polynomial Chaos Expansion")
        output = self.predict(X_test).T
        transposed_test = y_test.T 
        r2scores_in_time = np.asarray([r2_score(transposed_test[i], output[i]) for i in range(self.time_discretization)])
        return r2scores_in_time


class KarhunenLoeveMetamodel(object):
    """
    Karhunen-Loeve decomposition metamodel.
    """
    def __init__(
            self, variance_explained=0.99, 
            verbose=True
            )-> None:
        self.nb_modes = 5
        self.explained_variance_threshold = 0.9999
        self.verbose = verbose
        self.y_decomposed = False
        self.modes_fitted = False
        self.threshold = 1.0e-7

    def kl_output(self, y, simulation_time):

        self.y = y
        self.simulation_time = simulation_time

        vertices_number = simulation_time.shape[0]
        interval = ot.Interval(min(simulation_time), max(simulation_time))
        mesh = ot.IntervalMesher([vertices_number - 1]).build(interval)

        sample_size = y.shape[0]

        process_sample = ot.ProcessSample(mesh, sample_size, 1)
        process_sample.clear()
        for i in range(sample_size):
            process_sample.add(ot.Field(mesh, ot.Sample(y[i].reshape(1, -1).T)))
        self.process_sample = process_sample

        algo = ot.KarhunenLoeveSVDAlgorithm(process_sample, self.threshold)
        if self.verbose:
            print("Running Karhunen-Loeve decomposition")
        algo.run()

        self.kl_result = algo.getResult()

        self.time_discretization = len(self.simulation_time)

        self.eigenfunctions = self.kl_result.getScaledModesAsProcessSample()
        self.eigenvalues = self.kl_result.getEigenvalues()

        self.modes = np.asarray([[self.eigenfunctions.getSampleAtVertex(i)[j][0] for i in range(self.time_discretization)] for j in range(self.nb_modes)])

        self.y_decomposed = True

        eigval_square = np.asarray(self.eigenvalues)**2
        for i in range(self.eigenvalues.getDimension()):
            self.explained_variance = eigval_square[:i].sum()/eigval_square.sum()
            if self.explained_variance > self.explained_variance_threshold:
                self.nb_modes = i
                break
        if self.verbose:
            print(f"Explained variance for {i} modes is {self.explained_variance}")


    def fit(self, X, prior_mean='Constant', prior_kernel='AbsExp'):

        if prior_mean not in ['Constant', 'Linear', 'Quad']:
            raise ValueError(f"trend must be one of ['Constant', 'Linear', 'Quad']")

        if prior_kernel not in ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']:
            raise ValueError(f"kernel must be one of ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']")
        
        if self.y_decomposed:
            if self.nb_modes > self.eigenvalues.getDimension():
                raise ValueError("Number of modes must be less than the dimension of the Karhunen-Loeve decomposition")
            new_y = np.asarray(self.kl_result.project(self.process_sample))
        else:
            raise ValueError("You must first decompose the output data")
        
        self.X = X
        self.input_dimension = X.shape[1]
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, new_y, train_size=0.75, random_state=42)

        self.all_gps = []
        self.r2s = []

        for i in range(self.nb_modes): 
            gp = GpMetamodel(trend=prior_mean, kernel=prior_kernel, dimension=self.input_dimension)
            gp.fit(self.X_train, self.y_train[:, i])
            if self.verbose:
                print(f'Done fitting mode {i+1}')
            self.all_gps.append(gp)
            y_pred = gp.predict(self.X_test)
            r2_test = r2_score(self.y_test[:, i], y_pred)
            self.r2s.append(round(r2_test, 3))
            if self.verbose:
                print(f'Prediction accuracy for mode {i+1} is {r2_test}')

        _, _, _, self.y_test = train_test_split(self.X, self.y, train_size=0.75, random_state=42)

        self.modes_fitted = True

    def predict(self, X_new):
        if not self.modes_fitted:
            raise ValueError("You must first fit the modes of the Karhunen-Loeve decomposition")
        
        gp_pred = np.asarray([self.all_gps[i].predict(X_new) for i in range(self.nb_modes)]).T
        prediction = np.asarray([np.dot(gp_pred[i,:].ravel()*np.ones((self.time_discretization, 1)),
                                         self.modes[:self.nb_modes, :])[0,:] for i in range(X_new.shape[0])])
        
        return prediction
    
    def time_varying_r2score(self):
        if not self.modes_fitted:
            raise ValueError('You must first fit the modes of the Karhunen-Loeve')
        

class MLPMetamodel(object):
    """
    Artificial Neural Network metamodel.
    """
    def __init__(
            self, input_size=1, hidden_layer_size=10, output_size=1,
            solver='adam', alpha=0.0001, batch_size=100, 
            learning_rate='constant', learning_rate_init=0.001, 
            max_iter=200, verbose=True
            )-> None:
        self.hidden_layer_size = hidden_layer_size
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.trained_ = False

    def fit(self, X, y):

        class MLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.relu2 = nn.Sigmoid()
                self.fc3 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.fc3(x)
                return x

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.75, random_state=42)

        self.X_train, self.y_train = torch.tensor(self.X_train), torch.tensor(self.y_train)

        dataset = TensorDataset(self.X_train, self.y_train)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True) 

        # Initialize model, loss, and optimizer
        self.ann = MLP(self.input_size, self.hidden_layer_size, self.output_size)
        criterion = nn.MSELoss()  # Use CrossEntropyLoss for classification

        if self.solver == 'adam':
            optimizer = optim.Adam(self.ann.parameters(), lr=self.learning_rate_init)
        elif self.solver == 'sgd':
            optimizer = optim.SGD(self.ann.parameters(), lr=self.learning_rate_init)
        
        self.losses = []
        if self.verbose:
            print("Training Artificial Neural Network")
        for epoch in range(self.max_iter):
            for inputs, labels in data_loader:

                inputs = inputs.float()
                labels = labels.float() 
                # Forward pass
                outputs = self.ann(inputs)
                loss = criterion(outputs, labels)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.losses.append(loss.item())

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.max_iter}], Loss: {loss.item():.4f}")
                
        self.trained_ = True

    def predict(self, X_new):
        if not self.trained_:
            raise ValueError("You must first fit the Artificial Neural Network")
        y_pred = self.ann(X_new)
        return y_pred

    def r2_score(self):
        if not self.trained_:
            raise ValueError("You must first fit the Artificial Neural Network")
        output = self.predict(self.X_test).T
        transposed_test = self.y_test.T 
        r2scores_in_time = np