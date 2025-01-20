import openturns as ot
import numpy as np

class BayesCalibrationMCMC:
    def __init__(self, data: list, data_time_indices: list, metamodel: ot.Function, scaler=None, discrepancy=False) -> None:
        self.data = data
        self.data_time_indices = data_time_indices
        self.metamodel = metamodel
        self.discrepancy = discrepancy
        self.scaler = scaler


    class CalibrationFunction:
        def __init__(self, metamodel: ot.Function, data=None, scaler=None
        ) -> None:
            self.metamodel = metamodel
            self.data = data
            self.scaler = scaler

        def __call__(self, x: list) -> np.array:
            if self.metamodel.getInputDimension() == 1:
                return np.asarray(self.metamodel([x[0]]))
            else:
                data_times = []
                for i in range(len(self.data)):
                    data_times += list(self.data[i][:, 0]*24)
                data_times = np.asarray(sorted(data_times))
                return np.asarray(self.metamodel(self.scaler.transform(np.asarray([[x[0]] + [data_times[i]] for i in range(len(data_times))]))))

    class LikelihoodFunction:
        def __init__(self, data: np.array, data_time_indices: list, calib_function, discrepancy=False) -> None:
            self.data = data
            self.calibration_function = calib_function  
            self.data_time_indices = data_time_indices
            self.metamodel = self.calibration_function.metamodel
            self.discrepancy = discrepancy

        def __call__(self) -> ot.PythonFunction:
            # Define the log-likelihood function
            def log_likelihood(x) -> float:
                log_pdf = 0
                # Iterate over all data types
                for i in range(len(self.data)):
                    # Compute the difference between data and the calibration model output
                    model_output = self.calibration_function(x)
                    diff = self.data[i][:, 1] - model_output[self.data_time_indices[i]] 
                    # Compute the log-likelihood
                    log_pdf += -np.log(np.sum(diff**2)) * len(diff) / 2.0
                return [log_pdf]

            # Wrap the log-likelihood function in an OpenTURNS PythonFunction
            return ot.PythonFunction(1, 1, log_likelihood)
            #implement discrepancy ...
        
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
        proposal: ot.Distribution, burning: int, n_chains: int, sample_size: int
    ) -> list:
        self.support_min = support_min
        self.support_max = support_max
        self.proposal = proposal
        self.sample_size = sample_size
        self.burning = burning
        self.n_chains = n_chains

        self.support = ot.Interval(support_min, support_max)

        # Use uniform distribution as initial guess
        x_init = ot.Uniform(support_min[0], support_max[0]).getSample(n_chains)
        # Initialize calibration function
        calibration_function = self.CalibrationFunction(self.metamodel, self.data, self.scaler)
        # Initialize likelihood function
        likelihood_func = self.LikelihoodFunction(self.data, self.data_time_indices, calib_function=calibration_function)()

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