import openturns as ot
import numpy as np

def sample_positive_l1_sphere(p, n_samples):
    """Generate uniform samples from the (p-1)-simplex (positive L1 sphere)."""
    # Step 1: Sample p independent Exponential(1) variables
    Z = np.random.exponential(scale=1.0, size=(n_samples, p))
    # Step 2: Normalize to sum to 1
    X = Z / np.sum(Z, axis=1, keepdims=True)
    return X

class BayesCalibrationMCMC:
    def __init__(self, data: list, data_time_indices: list, metamodels: list, scalers=None, discrepancy=False) -> None:
        self.data = data
        self.data_time_indices = data_time_indices
        self.metamodels = metamodels
        self.discrepancy = discrepancy
        self.scalers = scalers

    class CalibrationFunction:
        def __init__(self, metamodel: ot.metamodel, data:list, data_time_indices: list, scaler=None
        ) -> None:
            self.metamodel = metamodel
            self.data = data
            self.data_time_indices = data_time_indices
            self.scaler = scaler

        def __call__(self, x: list) -> np.array:
            if self.metamodel.input_dimension == 1:
                time_indices = []
                for i in range(len(self.data_time_indices)):
                    time_indices += list(self.data_time_indices[i])
                time_indices = np.asarray(sorted(time_indices))
                if self.scaler is None:
                    return (np.asarray(self.metamodel.predict(np.asarray([x[0]]))).reshape(-1,1))[time_indices]
                else:
                    return ((self.metamodel.predict(self.scaler.transform(np.asarray([x[0]]).reshape(1, -1)))).reshape(-1,1))[time_indices]
            else:
                data_times = []
                for i in range(len(self.data)):
                    data_times += list(self.data[i][:, 0]*24)
                data_times = np.asarray(sorted(data_times))
                return self.metamodel.predict(self.scaler.transform(np.asarray([[x[0]] + [data_times[i]] for i in range(len(data_times))])))
            
    class LikelihoodFunction:
        def __init__(self, data: np.array, data_time_indices: list, calibration_functions: list, nb_weights: int, discrepancy=False
                     ) -> None:
            self.data = data
            self.calibration_functions = calibration_functions  
            self.data_time_indices = data_time_indices
            self.nb_weights = nb_weights
            self.discrepancy = discrepancy

        def __call__(self) -> ot.PythonFunction:
            # Define the log-likelihood function
            def log_likelihood(x) -> float:
                
                #w1+...+wp = 1 simplex :D :D :D
                dirichlet_weights = sample_positive_l1_sphere(len(self.calibration_functions), self.nb_weights)

                cal_funcs = ((np.asarray([np.asarray((self.calibration_functions)[i](x)) for i in range(len(self.calibration_functions))])).T)[0,:,:]
                
                # dot product of the weights and the calibration functions
                model_output = dirichlet_weights @ cal_funcs.T
                
                log_likelihoods = []
                for i in range(self.nb_weights):
                    log_pdf = 0
                    for j in range(len(self.data)):
                        diff = self.data[j][:, 1] - model_output[i][self.data_time_indices[j]] 
                        log_pdf += -np.log(np.sum(diff**2)) * len(diff) / 2.0
                    log_likelihoods.append(log_pdf)
                max_logs = np.max(np.asarray(log_likelihoods))

                # Compute the log-likelihood using logsumexp trick
                log_pdf = max_logs + np.log(np.sum(np.exp(np.asarray(log_likelihoods)) - max_logs)) - np.log(self.nb_weights)

                return [log_pdf]

            # Wrap the log-likelihood function in an OpenTURNS PythonFunction
            return ot.PythonFunction(1, 1, log_likelihood)
            #implement discrepancy 
        
    def gelman_rubin_test(
            self, samples: np.array, n_chains: int, sample_size: int
    ) -> float:
        # Compute the Gelman-Rubin convergence diagnostic
        sample_means = np.mean(samples, axis=1)
        sample_vars = np.var(samples, axis=1)
        B = sample_size / (n_chains - 1.) * sample_means.var(axis=0)
        W = sample_vars.mean(axis=0)
        V = (sample_size - 1.) / sample_size * W + (n_chains + 1.) / (sample_size * n_chains) * B
        return V/W

    def mcmc_run(
        self, support_min: list, support_max: list,
        proposal: ot.Distribution, burning: int, n_chains: int, sample_size: int, nb_weights: int
    ) -> list:
        self.support_min = support_min
        self.support_max = support_max
        self.proposal = proposal
        self.sample_size = sample_size
        self.burning = burning
        self.n_chains = n_chains
        self.nb_weights = nb_weights

        self.support = ot.Interval(support_min, support_max)

        # Use uniform distribution as initial guess
        x_init = ot.Uniform(support_min[0], support_max[0]).getSample(n_chains)
        # Initialize calibration function
        calibration_functions = [self.CalibrationFunction((self.metamodels)[i], self.data, self.data_time_indices, (self.scalers)[i]) for i in range(len(self.metamodels))]
        # Initialize likelihood function
        likelihood_func = self.LikelihoodFunction(self.data, self.data_time_indices, calibration_functions=calibration_functions, nb_weights=self.nb_weights)()

        samples = np.zeros((0, len(support_min)))
        gelman_rubin_test_sample = []

        for j in range(self.n_chains):
            print(f'Running Markov chain {j}')
            rwmh_sampler = ot.RandomWalkMetropolisHastings(likelihood_func, self.support, x_init[j], self.proposal)
            rwmh_sampler.setBurnIn(self.burning)
            chain_samples = rwmh_sampler.getSample(self.sample_size)
            samples = np.vstack((samples, chain_samples))
            gelman_rubin_test_sample.append(chain_samples)

        self.gelman_rubin = self.gelman_rubin_test(np.array(gelman_rubin_test_sample), n_chains=self.n_chains, sample_size=self.sample_size)
        print(f'Gelman-Rubin convergence diagnostic: {self.gelman_rubin[0]}')

        # Create the posterior distribution object
        self.posterior_distribution = ot.UserDefinedFactory().build(ot.Sample(samples))

        return self.posterior_distribution