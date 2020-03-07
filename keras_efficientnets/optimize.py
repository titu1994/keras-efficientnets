import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid

try:
    import inspect
    _inspect_available = True
except ImportError:
    _inspect_available = False

try:
    from joblib import Parallel, delayed
    _joblib_available = True
except ImportError:
    _joblib_available = False


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
    def compound_coeff(x, phi=phi, max_cost=max_cost):
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


def _sequential_optimize(param_grid, param_set, loss_func,
                         num_coeff, ineq_constraints, verbose):
    param_holder = np.empty((num_coeff,))

    for ix, param in enumerate(param_grid):
        # create a vector for the cost function and minimise using SLSQP
        for i in range(num_coeff):
            param_holder[i] = param[i]
        x0 = param_holder
        res = minimize(loss_func, x0, method='SLSQP', constraints=ineq_constraints)
        param_set[ix] = res.x

        if verbose:
            if (ix + 1) % 1000 == 0:
                print("Computed {:6d} parameter combinations...".format(ix + 1))

    return param_set


def _joblib_optimize(param, loss_func, num_coeff, ineq_constraints):
    x0 = np.asarray([param[i] for i in range(num_coeff)])
    res = minimize(loss_func, x0, method='SLSQP', constraints=ineq_constraints)
    return res.x


def optimize_coefficients(num_coeff=3, loss_func=None, phi=1.0, max_cost=2.0,
                          search_per_coeff=4, sort_by_loss=False, save_coeff=True,
                          tol=None, verbose=True):
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
        sort_by_loss: bool. Whether to sort the result set by its loss
            value, in order of lowest loss first.
        save_coeff: bool, whether to save the resulting coefficients
            into the file `param_coeff.npy` in current working dir.
        tol: float tolerance of error in the cost function. Used to
            select candidates which have a cost less than the tolerance.
        verbose: bool, whether to print messages during execution.

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
    if loss_func is None:
        loss_func = get_compound_coeff_func(phi, max_cost)

    # prepare inequality constraints
    ineq_constraints = {
        'type': 'ineq',
        'fun': lambda x: x - 1.
    }

    # Prepare a matrix to store results
    num_samples = search_per_coeff ** num_coeff
    param_range = [num_samples, num_coeff]

    # sorted by ParameterGrid acc to its key value, assuring sorted
    # behaviour for Python < 3.7.
    grid = {i: np.linspace(1.0, max_cost, num=search_per_coeff)
            for i in range(num_coeff)}

    if verbose:
        print("Preparing parameter grid...")
        print("Number of parameter combinations :", num_samples)

    param_grid = ParameterGrid(grid)

    if _joblib_available:
        with Parallel(n_jobs=-1, verbose=10 if verbose else 0) as parallel:
            param_set = parallel(delayed(_joblib_optimize)(param, loss_func, num_coeff, ineq_constraints)
                                 for param in param_grid)

        param_set = np.asarray(param_set)
    else:
        if verbose and num_samples > 1000:
            print("Consider using `joblib` library to speed up sequential "
                  "computation of {} combinations of parameters".format(num_samples))

        param_set = np.zeros(param_range)
        param_set = _sequential_optimize(param_grid, param_set, loss_func,
                                         num_coeff=num_coeff,
                                         ineq_constraints=ineq_constraints,
                                         verbose=verbose)

    # compute a minimum tolerance of the cost function
    # to select it in the candidate list.
    if tol is not None:
        if verbose:
            print("Filtering out samples below tolerance threshold...")

        tol = float(tol)
        cost_scores = np.asarray([loss_func(xi) for xi in param_set])
        param_set = param_set[np.where(cost_scores <= tol)]
    else:
        cost_scores = None

    # sort by lowest loss first
    if sort_by_loss:
        if verbose:
            print("Sorting by loss...")

        if cost_scores is None:
            cost_scores = ([loss_func(xi) for xi in param_set])
        else:
            cost_scores = cost_scores.tolist()

        cost_scores_id = [(idx, loss) for idx, loss in enumerate(cost_scores)]
        cost_scores_id = sorted(cost_scores_id, key=lambda x: x[1])

        ids = np.asarray([idx for idx, loss in cost_scores_id])
        # reorder the original param set
        param_set = param_set[ids, ...]

    if save_coeff:
        np.save('param_coeff.npy', param_set)

    return param_set


if __name__ == '__main__':

    def cost_func_wrapper(phi=1.0, max_cost=2.0):
        def cost_func(x: np.ndarray, phi=phi, max_cost=max_cost) -> float:
            depth = x[0] ** phi
            width = x[1] ** phi
            kernel_width = x[2] ** phi

            cost = (depth * width ** 2 * kernel_width ** 0.5)
            loss = (cost - max_cost) ** 2
            return loss
        return cost_func

    phi = 1.0
    loss_func = cost_func_wrapper(phi=phi, max_cost=2.0)

    results = optimize_coefficients(num_coeff=3, loss_func=loss_func,
                                    phi=1.0, max_cost=2.0, search_per_coeff=25,
                                    save_coeff=False, tol=None, sort_by_loss=True)

    print("Num unique configs = ", len(results))
    for i in range(10):  # print just the first 10 results out of 1000 results
        print(i + 1, results[i], "Cost :", loss_func(results[i]))

    phi = 4.0
    # params = [1.84, 1.007, 1.15]
    params = [1.04163396, 1.33328223, 1.16665207]
    cost = np.sqrt(loss_func(params, phi=phi, max_cost=0.))
    print("x0", params[0] ** phi)
    print("x1", params[1] ** phi)
    print("x2", params[2] ** phi)
    print(cost)
