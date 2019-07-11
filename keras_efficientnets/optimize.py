import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid


def get_compound_coeff_func(phi=1.0, max_cost=2.0):
    """
    Cost function from the EfficientNets paper
    to compute candidate values for alpha, beta
    and gamma parameters respectively.

    These values are then used to train models,
    and the validation accuracy is used to select
    the best base parameter set at phi = 1.

    # Arguments:
        phi: The base power of the parameters. Kept as 1
            for initial search of base parameters.
        max_cost: The maximum cost of permissible. User
            defined constant generally set to 2.

    # Returns:
        A function which accepts a numpy vector of 3 values,
        and computes the mean squared error between the
        `max_cost` value and the cost computed as
        `cost = x[0] * (x[1] ** 2) * (x[2] ** 2)`.

    # References:
        - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
    """
    def compound_coeff(x):
        depth = alpha = x[0]
        width = beta = x[1]
        resolution = gamma = x[2]

        # scale by power. Phi is generally kept as 1.0 during search.
        alpha = alpha ** phi
        beta = beta ** phi
        gamma = gamma ** phi

        # compute the cost function
        cost = alpha * (beta ** 2) * (gamma ** 2)
        return (cost - max_cost) ** 2

    return compound_coeff


def optimize_coefficients(num_coeff=3, cost_func=None, phi=1.0, max_cost=2.0,
                          search_per_coeff=4, save_coeff=True, tol=None):
    """
    Computes the possible values of any number of coefficients,
    given a cost function, phi and max cost permissible.

    Takes into account the search space per coefficient
    so that the subsequent grid search does not become
    prohibitively large.

    # Arguments:
        num_coeff: number of coefficients that must be optimized.
        cost_func: coefficient cost function that minimised to
            satisfy the least squares solution. The function can
            be user defined, in which case it must accept a numpy
            vector of length `num_coeff` defined above. It is
            suggested to use MSE against a pre-refined `max_cost`.
        phi: The base power of the parameters. Kept as 1
            for initial search of base parameters.
        max_cost: The maximum cost of permissible. User
            defined constant generally set to 2.
        search_per_coeff: int declaring the number of values tried
            per coefficient. Constructs a search space of size
            `search_per_coeff` ^ `num_coeff`.
        save_coeff: bool, whether to save the resulting coefficients
            into the file `param_coeff.npy` in current working dir.
        tol: float tolerance of error in the cost function. Used to
            select candidates which have a cost less than the tolerance.

    # Returns:
        A numpy array of shape [search_per_coeff ^ num_coeff, num_coeff],
        each row defining the value of the coefficients which minimise
        the cost function satisfactorily (to some machine precision).
    """
    phi = float(phi)
    max_cost = float(max_cost)
    search_per_coeff = int(search_per_coeff)

    # if user defined cost function is not provided, use the one from
    # the paper in reference.
    if cost_func is None:
        cost_func = get_compound_coeff_func(phi, max_cost)

    # prepare inequality constraints
    ineq_constraints = {
        'type': 'ineq',
        'fun': lambda x: x - 1.
    }

    # Prepare a matrix to store results
    param_range = [search_per_coeff ** num_coeff, num_coeff]
    param_set = np.zeros(param_range)

    # sorted by ParameterGrid acc to its key value, assuring sorted
    # behaviour for Python < 3.7.
    grid = {i: np.linspace(1.0, max_cost, num=search_per_coeff)
            for i in range(num_coeff)}

    param_grid = ParameterGrid(grid)
    for ix, param in enumerate(param_grid):
        # create a vector for the cost function and minimise using SLSQP
        x0 = np.array([param[i] for i in range(num_coeff)])
        res = minimize(cost_func, x0, method='SLSQP', constraints=ineq_constraints)
        param_set[ix] = res.x

    # compute a minimum tolerance of the cost function
    # to select it in the candidate list.
    if tol is not None:
        tol = float(tol)
        cost_scores = np.array([cost_func(xi) for xi in param_set])
        param_set = param_set[np.where(cost_scores <= tol)]

    if save_coeff:
        np.save('param_coeff.npy', param_set)

    return param_set
