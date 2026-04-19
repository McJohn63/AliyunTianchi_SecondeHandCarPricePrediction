import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

def find_best_weights(y_true, oof_list):
    def objective(weights):
        combined_oof = sum(w * oof for w, oof in zip(weights, oof_list))
        return mean_absolute_error(np.expm1(y_true), np.expm1(combined_oof))

    initial_weights = np.array([0.5, 0.5])
    bounds = [(0, 1)] * 2
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    res = minimize(objective, initial_weights, method='SLSQP', 
                   bounds=bounds, constraints=constraints)
    
    return res.x