import numpy as np
import openturns as ot


class HSIC:
    """
    Class to compute the Hilbert-Schmidt Independence Criterion (HSIC) and its associated asymptotic p-values with the help of OpenTURNS.
    """
    def __init__(self, X: ot.Sample, y: ot.Sample) -> None:
        self.X = X
        self.y = y

    def r2_hsic(self) -> list:
        estimator = ot.HSICUStat()
        d, N = self.X.getDimension(), self.y.getDimension()
        hsic_vals, p_vals = [], []
        cov_inputs = []
        for j in range(d):
            #use the standard deviation of the input data to set the scale of the covariance function
            cov = ot.SquaredExponential(1)
            cov.setScale(self.X[:, j].computeStandardDeviation())
            cov_inputs.append(cov)
        if N > 1:
            for i in range(N):
                cov_total = cov_inputs.copy()
                cov_total.append(ot.SquaredExponential(self.y[:, i].computeStandardDeviation()))
                hsic_comp = ot.HSICEstimatorGlobalSensitivity(cov_total, self.X, self.y[:, i], estimator)
                hsic_vals.append(list(np.abs(hsic_comp.getR2HSICIndices())))
                p_vals.append(list(hsic_comp.getPValuesAsymptotic()))
        else:
            cov_total = cov_inputs
            cov_total.append(ot.SquaredExponential(self.y.computeStandardDeviation()))
            hsic_comp = ot.HSICEstimatorGlobalSensitivity(cov_total, self.X, self.y, estimator)
            hsic_vals.append(list(np.abs(hsic_comp.getR2HSICIndices())))
            p_vals.append(list(hsic_comp.getPValuesAsymptotic()))

        hsic_vals = np.asarray(list(np.transpose(np.asarray(hsic_vals))))
        p_vals = np.transpose(np.asarray(p_vals))
        
        return hsic_vals, p_vals
    

    def target_r2_hsic(self, target) -> list:

        dist = ot.DistanceToDomainFunction(ot.Interval(target, float('inf')))

        estimator = ot.HSICVStat() 
        d, N = self.X.getDimension(), self.y.getDimension()
        hsic_vals, p_vals = [], []
        cov_inputs = []

        for j in range(d):
            #use the standard deviation of the input data to set the scale of the covariance function
            cov = ot.SquaredExponential(1)
            cov.setScale(self.X[:, j].computeStandardDeviation())
            cov_inputs.append(cov)
        if N > 1:
            for i in range(N):
                sigma = self.y[:,i].computeStandardDeviation()
                func = ot.SymbolicFunction('x', 'exp(- (5 * x)/{0} )'.format(sigma))
                weight = ot.ComposedFunction(func, dist)

                cov_total = cov_inputs.copy()

                cov_total.append(ot.SquaredExponential(self.y[:, i].computeStandardDeviation()))
                hsic_comp = ot.HSICEstimatorTargetSensitivity(cov_total, self.X, self.y[:, i], estimator, weight)

                hsic_vals.append(list(np.abs(hsic_comp.getR2HSICIndices())))

                hsic_comp.setPermutationSize(100)

                p_vals.append(list(hsic_comp.getPValuesPermutation()))

                #find a way to compute the exact number of samples selected by the filter
        else:
            sigma = self.y.computeStandardDeviation()
            func = ot.SymbolicFunction('x', 'exp(- (5 * x)/{0} )'.format(sigma))
            weight = ot.ComposedFunction(func, dist)

            cov_total = cov_inputs.copy()
            cov_total.append(ot.SquaredExponential(self.y.computeStandardDeviation()))

            hsic_comp = ot.HSICEstimatorTargetSensitivity(cov_total, self.X, self.y, estimator, weight)
            hsic_vals.append(list(np.abs(hsic_comp.getR2HSICIndices())))
            hsic_comp.setPermutationSize(100)
            p_vals.append(list(hsic_comp.getPValuesPermutation()))

        hsic_vals = np.asarray(list(np.transpose(np.asarray(hsic_vals))))
        p_vals = np.transpose(np.asarray(p_vals))

        return hsic_vals, p_vals
    
    def conditional_r2_hsic(self, target) -> list:
        
        dist = ot.DistanceToDomainFunction(ot.Interval(target, float('inf')))

        d, N = self.X.getDimension(), self.y.getDimension()
        hsic_vals, p_vals = [], []
        cov_inputs = []

        for j in range(d):
            #use the standard deviation of the input data to set the scale of the covariance function
            cov = ot.SquaredExponential(1)
            cov.setScale(self.X[:, j].computeStandardDeviation())
            cov_inputs.append(cov)
        if N > 1:
            for i in range(N):
                sigma = self.y[:,i].computeStandardDeviation()
                func = ot.SymbolicFunction('x', 'exp(- (5 * x)/{0} )'.format(sigma))
                weight = ot.ComposedFunction(func, dist)

                cov_total = cov_inputs.copy()

                cov_total.append(ot.SquaredExponential(self.y[:, i].computeStandardDeviation()))
                hsic_comp = ot.HSICEstimatorConditionalSensitivity(cov_total, self.X, self.y[:, i], weight)

                hsic_vals.append(list(np.abs(hsic_comp.getR2HSICIndices())))

                hsic_comp.setPermutationSize(100)

                p_vals.append(list(hsic_comp.getPValuesPermutation()))

                #find a way to compute the exact number of samples selected by the filter
        else:
            sigma = self.y.computeStandardDeviation()
            func = ot.SymbolicFunction('x', 'exp(- (5 * x)/{0} )'.format(sigma))
            weight = ot.ComposedFunction(func, dist)

            cov_total = cov_inputs.copy()
            cov_total.append(ot.SquaredExponential(self.y.computeStandardDeviation()))

            hsic_comp = ot.HSICEstimatorConditionalSensitivity(cov_total, self.X, self.y, weight)
            hsic_vals.append(list(np.abs(hsic_comp.getR2HSICIndices())))
            hsic_comp.setPermutationSize(100)
            p_vals.append(list(hsic_comp.getPValuesPermutation()))
        
        hsic_vals = np.asarray(list(np.transpose(np.asarray(hsic_vals))))
        p_vals = np.transpose(np.asarray(p_vals))

        return hsic_vals, p_vals
