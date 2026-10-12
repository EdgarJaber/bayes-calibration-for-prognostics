import numpy as np
import openturns as ot
import openturns.experimental as otexp


def _ot_class(name):
    """Resolve an OpenTURNS class that moved between the core and the
    experimental module across versions. In OT 1.24-1.26 the Gaussian-process
    classes live in openturns.experimental; from 1.27 they are in the core
    openturns namespace. Prefer the core, fall back to experimental."""
    if hasattr(ot, name):
        return getattr(ot, name)
    if hasattr(otexp, name):
        return getattr(otexp, name)
    raise AttributeError(
        f"OpenTURNS {ot.__version__} exposes neither ot.{name} nor otexp.{name}."
    )


_GaussianProcessFitter = _ot_class("GaussianProcessFitter")
_GaussianProcessRegression = _ot_class("GaussianProcessRegression")
_GaussianProcessConditionalCovariance = _ot_class("GaussianProcessConditionalCovariance")

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


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

        # Initialize and run the GP model (new OpenTURNS API:
        # GaussianProcessFitter calibrates hyperparameters, then
        # GaussianProcessRegression conditions the process on the data).
        fitter = _GaussianProcessFitter(
            ot.Sample(X_train),
            ot.Sample(y_train.reshape(-1, 1)),
            covarianceModel, basis
        )
        fitter.run()
        self.gp_ = _GaussianProcessRegression(fitter.getResult())
        self.gp_.run()

        self.gp_result_ = self.gp_.getResult()
        self.gp = self.gp_result_.getMetaModel()

        self.trained_ = True


    def predict(self, X_test, return_std=False):
        if not self.trained_:
            raise ValueError("The model has not been trained yet.")

        y_pred = self.gp(X_test)

        if not return_std:
            return np.array(y_pred).reshape(-1, 1)

        # Predictive std via the conditional covariance (new API replacement for
        # the old callable KrigingResult(...).getStandardDeviation()).
        gccc = _GaussianProcessConditionalCovariance(self.gp_result_)
        var = np.array(gccc.getConditionalMarginalVariance(ot.Sample(X_test)))
        y_std = np.sqrt(np.maximum(var, 0.0))
        return np.array(y_pred), y_std
    
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

        # Initialize and run the GP model (new OpenTURNS API)
        fitter = _GaussianProcessFitter(
            ot.Sample(X_train),
            ot.Sample(y_train.reshape(-1, 1)),
            covarianceModel, basis
        )
        fitter.run()
        self.gp_ = _GaussianProcessRegression(fitter.getResult())
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
            explained_variance_threshold=0.99, 
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

            if self.eigenvalues.getDimension() == 1:
                self.nb_modes = 1
                self.explained_variance = 1.0
            else:
                dim = self.eigenvalues.getDimension()
                # Fall back to every retained eigenvalue if the threshold is never
                # reached -- checking i in range(dim) only ever sums at most
                # dim - 1 eigenvalues, so it can undershoot explained_variance=1.0
                # by the last eigenvalue's share and never break, leaving nb_modes
                # at its stale __init__ default (larger than the number of modes
                # KarhunenLoeveSVDAlgorithm actually retained -> IndexError below).
                self.nb_modes = dim
                self.explained_variance = 1.0
                for i in range(1, dim + 1):
                    self.explained_variance = eigval_square[:i].sum()/eigval_square.sum()
                    if self.explained_variance >= self.explained_variance_threshold:
                        self.nb_modes = i
                        break
            
            if self.verbose:
                print(f"Explained variance for {self.nb_modes} modes is {self.explained_variance}")
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


class StochasticCollocGPMetamodel(BaseEstimator):
    """
    Wrapper for OpenTURNS Gaussian Process.
    """
    def __init__(
        self, trend: str, kernel: str, 
        input_dimension: int, simulation_time: np.array, 
        scaler: StandardScaler,
        noise: float = None
    ) -> None:
        self.trend = trend
        self.kernel = kernel
        self.input_dimension = input_dimension
        self.scaler = scaler
        self.noise = noise
        self.simulation_time = simulation_time
        self.trained_ = False
        self.X_train_ = None 
        self.y_train_ = None

    def fit(self, X_train, y_train):

        gp = GpMetamodel(self.trend, self.kernel, self.input_dimension, self.noise)
        gp.fit(X_train, y_train)

        self.X_train_ = X_train
        self.y_train_ = y_train

        self.gp_ = gp
        self.trained_ = True

    def predict(self, X_test, return_std=False):
        
        if not self.trained_:
            raise ValueError("The model has not been trained yet.")

        y_pred, self.X_test_scaled = [], []
        for l in range(X_test.shape[0]):
            rand_test = []
            for q in range(0, len(self.simulation_time)):
                rand_test.append(list(X_test[l, :self.input_dimension-1]) + [self.simulation_time[q]*24])
            scaled_test = self.scaler.transform(rand_test)
            self.X_test_scaled.append(scaled_test)
            y_pred.append(self.gp_.predict(scaled_test, return_std=return_std))
        y_pred = np.array(y_pred)
        self.X_test_scaled = np.array(self.X_test_scaled)

        return y_pred


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