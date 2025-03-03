import numpy as np
import openturns as ot

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import DataLoader, TensorDataset


class GpMetamodel(BaseEstimator):
    """
    Wrapper for OpenTURNS Gaussian Process.
    """
    def __init__(
        self, trend: str, kernel: str, 
        input_dimension: int, noise: float = None
    ) -> None:
        self.trend = trend
        self.kernel = kernel
        self.input_dimension = input_dimension
        self.noise = noise
        self.trained_ = False
        self.X_train_ = None 
        self.y_train_ = None 

    def fit(self, X_train, y_train):

        # Validate inputs
        if self.trend not in ['Constant', 'Linear', 'Quad']:
            raise ValueError(f"trend must be one of ['Constant', 'Linear', 'Quad']")

        if self.kernel not in ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']:
            raise ValueError(f"kernel must be one of ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']")

        # Save training data for later use
        self.X_train_ = X_train
        self.y_train_ = y_train

        # Initialize basis and kernel
        if self.trend == 'Constant':
            basis = ot.ConstantBasisFactory(self.input_dimension).build()
        elif self.trend == 'Linear':
            basis = ot.LinearBasisFactory(self.input_dimension).build()
        elif self.trend == 'Quad':
            basis = ot.QuadraticBasisFactory(self.input_dimension).build()

        if self.kernel == 'AbsExp':
            covarianceModel = ot.AbsoluteExponential([1.0] * self.input_dimension)
        elif self.kernel == 'SqExp':
            covarianceModel = ot.SquaredExponential([1.0] * self.input_dimension)
        elif self.kernel == 'M-1/2':
            covarianceModel = ot.MaternModel([1.0] * self.input_dimension, [1.0], 0.5)
        elif self.kernel == 'M-3/2':
            covarianceModel = ot.MaternModel([1.0] * self.input_dimension, [1.0], 1.5)
        elif self.kernel == 'M-5/2':
            covarianceModel = ot.MaternModel([1.0] * self.input_dimension, [1.0], 2.5)

        if self.noise:
            covarianceModel.setNuggetFactor(self.noise)

        # Initialize and run the GP model
        self.gp_ = ot.KrigingAlgorithm(
            ot.Sample(X_train),
            ot.Sample(y_train.reshape(-1, 1)),
            covarianceModel, basis
        )
        self.gp_.run()

        self.gp = self.gp_.getResult().getMetaModel()

        self.trained_ = True


    def predict(self, X_test):
        if not self.trained_:
            raise ValueError("The model has not been trained yet.")
        
        y_pred = self.gp(X_test)

        return np.array(y_pred)

    def __sklearn_is_fitted__(self):
        return self.trained_

    def __getstate__(self):
        state = self.__dict__.copy()
        state['gp_'] = None  # Exclude the non-picklable `self.gp` object
        state['gp'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Recreate the `self.gp` object if training data is available
        if self.X_train_ is not None and self.y_train_ is not None:
            self.fit(self.X_train_, self.y_train_)



class GpMetamodelInd(BaseEstimator):
    """
    Wrapper for OpenTURNS Gaussian Process.
    """
    def __init__(
        self, trend: str, kernel: str, 
        input_dimension: int, noise: float = None, index = None
    ) -> None:
        self.trend = trend
        self.kernel = kernel
        self.input_dimension = input_dimension
        self.noise = noise
        self.trained_ = False
        self.X_train_ = None 
        self.y_train_ = None 
        self.index = index

    def fit(self, X_train, y_train):

        # Validate inputs
        if self.trend not in ['Constant', 'Linear', 'Quad']:
            raise ValueError(f"trend must be one of ['Constant', 'Linear', 'Quad']")

        if self.kernel not in ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']:
            raise ValueError(f"kernel must be one of ['AbsExp', 'SqExp', 'M-1/2', 'M-3/2', 'M-5/2']")

        # Save training data for later use
        self.X_train_ = X_train
        self.y_train_ = y_train

        # Initialize basis and kernel
        if self.trend == 'Constant':
            basis = ot.ConstantBasisFactory(self.input_dimension).build()
        elif self.trend == 'Linear':
            basis = ot.LinearBasisFactory(self.input_dimension).build()
        elif self.trend == 'Quad':
            basis = ot.QuadraticBasisFactory(self.input_dimension).build()

        if self.kernel == 'AbsExp':
            covarianceModel = ot.AbsoluteExponential([1.0] * self.input_dimension)
        elif self.kernel == 'SqExp':
            covarianceModel = ot.SquaredExponential([1.0] * self.input_dimension)
        elif self.kernel == 'M-1/2':
            covarianceModel = ot.MaternModel([1.0] * self.input_dimension, [1.0], 0.5)
        elif self.kernel == 'M-3/2':
            covarianceModel = ot.MaternModel([1.0] * self.input_dimension, [1.0], 1.5)
        elif self.kernel == 'M-5/2':
            covarianceModel = ot.MaternModel([1.0] * self.input_dimension, [1.0], 2.5)

        if self.noise:
            covarianceModel.setNuggetFactor(self.noise)

        # Initialize and run the GP model
        self.gp_ = ot.KrigingAlgorithm(
            ot.Sample(X_train),
            ot.Sample(y_train.reshape(-1, 1)),
            covarianceModel, basis
        )
        self.gp_.run()

        self.gp = self.gp_.getResult().getMetaModel()

        self.trained_ = True

    def predict(self, X_test):
        if not self.trained_:
            raise ValueError("The model has not been trained yet.")
        if self.index != None:
            X_ = self.X_train_.mean(axis=0)
            X_ = np.tile(X_, (X_test.shape[0], 1))
            X_[:, self.index] = X_test
            X_test = X_
            y_pred = self.gp(X_test)
        else:
            y_pred = self.gp(X_test)
        return np.array(y_pred)

    def __sklearn_is_fitted__(self):
        return self.trained_

    def __getstate__(self):
        state = self.__dict__.copy()
        state['gp_'] = None  # Exclude the non-picklable `self.gp` object
        state['gp'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Recreate the `self.gp` object if training data is available
        if self.X_train_ is not None and self.y_train_ is not None:
            self.fit(self.X_train_, self.y_train_)





class VPCEMetamodel(BaseEstimator):
    """
    Vector Polynomial chaos expansions with regression method for coefficient estimation.
    Comes with a fit and predict method.
    """
    def __init__(
            self, degree: int, q_norm: float, 
            input_dimension: int, prior_distribution_list: list,
            verbose=True
            )-> None:
        self.degree = degree
        self.q_norm = q_norm
        self.verbose = verbose
        #self.prior = prior
        self.prior_distribution_list = prior_distribution_list
        self.input_dimension = input_dimension
        self.trained_ = False
        self.X_train_ = None 
        self.y_train_ = None 

    def fit(self, X_train, y_train):
        # Save training data for later use
        self.X_train_ = X_train
        self.y_train_ = y_train

        self.distribution = ot.ComposedDistribution(self.prior_distribution_list)
        
        multivar_basis = ot.OrthogonalProductPolynomialFactory(self.prior_distribution_list)
        selection_algo = ot.LeastSquaresMetaModelSelectionFactory()
        projection_strategy = ot.LeastSquaresStrategy(X_train, y_train, selection_algo)
        enum_func = ot.HyperbolicAnisotropicEnumerateFunction(self.input_dimension, self.q_norm)
        P = enum_func.getBasisSizeFromTotalDegree(self.degree)
        adaptive_strategy = ot.FixedStrategy(multivar_basis, P)
        self.chaos_algo = ot.FunctionalChaosAlgorithm(X_train, y_train, self.distribution, adaptive_strategy, projection_strategy)
        self.chaos_algo.run()
        if self.verbose:
            print(f"Running Polynomial Chaos Expansion with regression method for degree {self.degree}, q-norm {self.q_norm}")
        self.pce = self.chaos_algo.getResult().getMetaModel()
        self.trained_ = True

    def predict(self, X_test):
        if not self.trained_:
            raise ValueError("You must first fit the Polynomial Chaos Expansion")
        #X_test = StandardScaler().fit_transform(X_test)
        y_pred = self.pce(X_test)
        return y_pred
    
    def r2_score(self, X_test, y_test, time_discretization):
        if not self.trained_:
            raise ValueError("You must first fit the Polynomial Chaos Expansion")
        output = np.asarray(self.predict(X_test)).T
        transposed_test = y_test.T 
        r2scores_in_time = np.asarray([r2_score(transposed_test[i], output[i]) for i in range(len(time_discretization))])
        return r2scores_in_time
    
    def __sklearn_is_fitted__(self):
        return self.trained_

    def __getstate__(self):
        state = self.__dict__.copy()
        state['pce'] = None  # Exclude the non-picklable `self.pce` object
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Recreate the `self.gp` object if training data is available
        if self.X_train_ is not None and self.y_train_ is not None:
            self.fit(self.X_train_, self.y_train_)


class KarhunenLoeveMetamodel(BaseEstimator):
    """
    Karhunen-Loeve decomposition metamodel.
    """
    def __init__(
            self, metamodel: object, input_dimension: int, simulation_time: np.array,
            explained_variance_threshold=0.9999, 
            verbose=True
            )-> None:
        self.metamodel = metamodel
        self.input_dimension = input_dimension
        self.simulation_time = simulation_time
        self.nb_modes = 10
        self.explained_variance_threshold = explained_variance_threshold
        self.verbose = verbose
        self.y_decomposed = False
        self.trained_ = False
        self.threshold = 1.0e-7
        self.X_train = None 
        self.y_train = None

    class KLResult:
        def __init__(self, simulation_time:np.array, threshold=1.0e-7, nb_modes=10, verbose=True, explained_variance=0.99
        ) -> None:
            self.simulation_time = simulation_time
            self.threshold = threshold
            self.verbose = verbose
            self.nb_modes = nb_modes
            self.explained_variance_threshold = explained_variance

        def __call__(self, y: np.array) -> np.array:

            vertices_number = self.simulation_time.shape[0]
            interval = ot.Interval(min(self.simulation_time), max(self.simulation_time))
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

            self.kl_algo_result = algo.getResult()

            self.time_discretization = len(self.simulation_time)

            self.eigenfunctions = self.kl_algo_result.getScaledModesAsProcessSample()
            self.eigenvalues = self.kl_algo_result.getEigenvalues()

            eigval_square = np.asarray(self.eigenvalues)**2
            for i in range(self.eigenvalues.getDimension()):
                self.explained_variance = eigval_square[:i].sum()/eigval_square.sum()
                if self.explained_variance > self.explained_variance_threshold:
                    self.nb_modes = i
                    break
            if self.verbose:
                print(f"Explained variance for {i} modes is {self.explained_variance}")

            self.modes = np.asarray([[self.eigenfunctions.getSampleAtVertex(i)[j][0] for i in range(self.time_discretization)] for j in range(self.nb_modes)])

            self.y_decomposed = True

            return self.modes


    def fit(self, X_train, y_train):
        
        self.X_train_ = X_train
        self.y_train_ = y_train

        self.kl_result = self.KLResult(self.simulation_time, self.threshold, nb_modes=self.nb_modes, verbose=self.verbose, explained_variance=self.explained_variance_threshold)
        self.modes = self.kl_result(y_train)

        y_train = np.asarray(self.kl_result.kl_algo_result.project(self.kl_result.process_sample))
            
        self.all_metamodels = []
        self.r2s = []

        for i in range(self.kl_result.nb_modes): 
            mm = clone(self.metamodel)
            mm.fit(X_train, y_train[:, i].reshape(-1,1))            
            if self.verbose:
                print(f'Done fitting mode {i+1}')
            self.all_metamodels.append(mm)

        self.trained_ = True

    def predict(self, X_test):
        if not self.trained_:
            raise ValueError("You must first fit the modes of the Karhunen-Loeve decomposition")
        mm_pred = np.asarray([self.all_metamodels[i].predict(X_test) for i in range(self.kl_result.nb_modes)])[:,:,0].T
        prediction = np.asarray([np.dot(mm_pred[i,:].ravel()*np.ones((self.kl_result.time_discretization, 1)),self.modes[:self.kl_result.nb_modes, :])[0,:] for i in range(X_test.shape[0])])
        return prediction
        

    def r2_score(self, X_test, y_test, simulation_time):
        if not self.trained_:
            raise ValueError("You must first fit the Karhunen-Loeve decomposition")
        output = np.asarray(self.predict(X_test)).T
        transposed_test = y_test.T 
        r2scores_in_time = np.asarray([r2_score(transposed_test[i], output[i]) for i in range(len(simulation_time))])
        return r2scores_in_time

    def __sklearn_is_fitted__(self):
        return self.trained_

    def __getstate__(self):
        state = self.__dict__.copy()
        state['mm'] = None  # Exclude the non-picklable `self.mm` object
        state['all_metamodels'] = None
        state['kl_result'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Recreate the `self.gp` object if training data is available
        if self.X_train_ is not None and self.y_train_ is not None:
            self.fit(self.X_train_, self.y_train_)

        

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
