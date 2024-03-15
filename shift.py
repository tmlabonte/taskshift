from argparse import Namespace
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import mean_squared_error

plt.rcParams["text.usetex"] = True

d_end = 505
d_start = 5
d_step = 5
n = 100
n_test = 100
trials = 10
#var = np.sqrt(2)
var = 1

# cov_types = ["isotropic", "spiked", "poly"]
cov_types = ["spiked"]
# poly_exponents = [0.5, 1, 2]
poly_exponents = [2]
# spiked_sizes = [100, "d", "d^2"]
spiked_sizes = ["d"]
spiked_nums = [1]
# theta_star_types = ["unif", "e1"]
# temperatures = [0.01, 0.001, 0.0001]
theta_star_types = ["e1"]
# y_types = ["sgn", "gaussian"]
y_types = ["sgn"]

args = Namespace()
ds = np.arange(d_start, d_end, d_step)
rng = np.random.default_rng()

def repeated_argmax(x, num):
    y = np.zeros(len(x))
    argmaxes = np.argpartition(x, -num)[-num:]
    y[argmaxes] = 1
    return y

def set_display_names(args, fname):
    # Adds y_type to filename.
    if args.y_type == "sgn":
        fname += "_sgn"
        display_y_type = r"$y=sgn(\tilde{y})$"
    elif args.y_type == "gaussian":
        fname += "_gaussian"
        display_y_type = r"$y\sim\mathcal{N}(0,1)$"
    else:
        raise ValueError()

    # Adds theta_star_type to filename.
    if args.theta_star_type == "unif":
        fname += "_unif"
        display_theta_star_type = r"$\theta^\star\sim U(-1,1)$"
    elif args.theta_star_type == "e1":
        fname += "_e1"
        display_theta_star_type = r"$\theta^\star=e_1$"
    elif args.theta_star_type == "e12":
        fname += "_e12"
        display_theta_star_type = r"$\theta^\star=e_{12}$"
    else:
        raise ValueError()

    # Adds cov_type to filename.
    if args.cov_type == "isotropic":
        fname += "_isotropic.png"
    elif args.cov_type == "spiked":
        fname += "_spiked.png"
    elif args.cov_type == "poly":
        fname += "_poly.png"
    else:
        raise ValueError()

    return fname, display_theta_star_type, display_y_type

def plot(args, results):
    # Defines a helper function for plotting the results.
    def plot_helper(
        names, display_names, fname, xmin=None, xmax=None, ymin=0, ymax=None
    ):
        fname, display_theta_star_type, display_y_type = \
                set_display_names(args, fname)

        start = (xmin - d_start) // d_step if xmin else 0
        end = (xmax - d_start) // d_step if xmax else len(ds)

        # Plots specified range of data.
        for name, display_name in zip(names, display_names):
            for j, result in enumerate(results):
                if args.cov_type == "spiked":
                    for k, sub_result in enumerate(result):
                        label = display_name
                        label += rf" size=${spiked_sizes[j]}$,"
                        label += rf" num=${spiked_nums[k]}$"
                        plt.plot(
                            ds[start:end],
                            sub_result[name][start:end],
                            label=label,
                        )
                else:
                    if args.cov_type == "isotropic":
                        label = display_name
                    elif args.cov_type == "poly":
                        label = display_name
                        label += rf" poly=${poly_exponents[j]}$"
                    else:
                        raise ValueError()

                    plt.plot(
                        ds[start:end],
                        result[name][start:end],
                        label=label,
                    )

        # Automatically crops the y-axis if a bound on the x-axis is specified.
        if (xmin or xmax) and not ymax:
            if args.cov_type == "spiked":
                ymax = max(np.asarray([
                    sub_result[name][start:end] for name in names
                    for result in results for sub_result in result
                ]).flatten())
            else:
                ymax = max(np.asarray([
                    result[name][start:end] for name in names
                    for result in results
                ]).flatten())

        title = fr"{display_theta_star_type}, {display_y_type}, "
        title += fr"{args.cov_type}, $n={n}$, "
        title += fr"$\|\theta^\star\|_2^2={round(var, 3)}$, {trials} trials"

        plt.ylim(ymin, ymax)
        plt.legend()
        plt.title(title)
        plt.xlabel("d")
        plt.savefig(fname, bbox_inches="tight", dpi=600)
        plt.clf()

    # Updates results with theoretical prediction term.
    for result in results:
        if args.cov_type == "spiked":
            for sub_result in result:
                sub_result["n / d * var"] = n * (1 / ds) * var
        else:
            result["n / d * var"] = n * (1 / ds) * var

    # Sets display parameters.
    risk_diff_display_name = r"$L(\hat{\theta})-L(\tilde{\theta})$"
    ymax = 2 * var

    # Plots the results.
    plot_helper(
        ["risk_diff"],
        [risk_diff_display_name],
        "diff",
        ymax=ymax,
    )
    plot_helper(
        ["risk_diff", "n / d * var"],
        [risk_diff_display_name, r"$\frac{n}{d}\|\theta^\star\|_2^2$"],
        "pred",
        xmin=n + 10,
    )
    plot_helper(
        ["theta_tilde_risk", "theta_hat_risk", "theta_new_risk"],
        [r"$L(\tilde{\theta})$", r"$L(\hat{\theta})$", r"$L(\hat{\theta}^\prime)$"],
         "risk",
         ymax=ymax,
    )
    plot_helper(
        ["theta_tilde_norm", "theta_hat_norm", "theta_new_norm"],
        [r"$\|\tilde{\theta}\|_2$", r"$\|\hat{\theta}\|_2$", r"$\|\hat{\theta}^\prime\|_2$"],
        "norm",
        ymax=ymax,
    )

def experiment_trial(args, results, idx, d):
    # Generates the ground-truth regressor by a uniform distribution and scales
    # it to have the appropriate norm.
    if args.theta_star_type == "unif":
        theta_star = rng.uniform(-1, 1, d)
        theta_star = np.sqrt(var) * (theta_star / np.linalg.norm(theta_star))
    elif args.theta_star_type == "e1":
        theta_star = np.zeros(d)
        theta_star[0] = var
    elif args.theta_star_type == "e12":
        theta_star = np.zeros(d)
        theta_star[0] = var
        theta_star[1] = var
    else:
        raise ValueError()

    """
    elif args.cov_type == "bilevel":
        cov = np.identity(d)
        p = np.log(d) / np.log(n)
        r = 0.5
        a = n ** -(p-r-0.01)
        s = n ** r
        for j in range(ceil(s)):
            cov[j, j] = (a * d) / s
        for j in range(ceil(s), d):
            cov[j, j] = ((1-a)*d)/(d-s)
    """

    # Generates the train data and labels using the ground-truth regressor.
    if args.cov_type == "isotropic":
        cov = np.identity(d)
    elif args.cov_type == "spiked":
        cov = np.identity(d)
        for j in range(min(args.spiked_num, d)):
            if args.spiked_size == "d^2":
                cov[j, j] = d ** 2
            elif args.spiked_size == "d":
                cov[j, j] = d
            else:
                cov[j, j] = args.spiked_size

        # Rescales covariance to have norm 1.
        cov = cov / cov[0, 0]
    elif args.cov_type == "poly":
        cov = np.diag([(j + 1) ** -args.poly_exponent for j in range(d)])
        # TODO: rescale to have norm 1?
    else:
        raise ValueError()

    X = rng.multivariate_normal(np.zeros(d), cov, n).T
    y_tilde = X.T @ theta_star

    # Generates classifier data as sign of the regression labels or as Gaussian.
    if args.y_type == "sgn":
        y = np.sign(y_tilde)
        # y *= np.mean(np.abs(y_tilde))
    elif args.y_type == "gaussian":
        y = rng.multivariate_normal(np.zeros(n), np.identity(n))
    else:
        raise ValueError()

    # Generates the test data and labels using the ground-truth regressor.
    X_test = rng.multivariate_normal(np.zeros(d), np.identity(d), n_test).T
    y_tilde_test = X_test.T @ theta_star

    # Computes the minimum-norm interpolators for regression and classification.
    M = X.T @ np.linalg.pinv(X @ X.T, hermitian=True)
    theta_tilde = M.T @ y_tilde
    theta_hat = M.T @ y

    # Computes the test risk of the minimum-norm interpolators.
    theta_tilde_test_risk = mean_squared_error(
        X_test.T @ theta_tilde, y_tilde_test)
    theta_hat_test_risk = mean_squared_error(
        X_test.T @ theta_hat, y_tilde_test)
    theta_tilde_norm = np.linalg.norm(theta_tilde)
    theta_hat_norm = np.linalg.norm(theta_hat)

    # Computes the difference of the test risks of the classifier and regressor.
    risk_diff = (theta_hat_test_risk - theta_tilde_test_risk)

    # Adds relevant metrics to the results dictionary.
    results["risk_diff"][idx].append(risk_diff)
    results["theta_tilde_risk"][idx].append(theta_tilde_test_risk)
    results["theta_hat_risk"][idx].append(theta_hat_test_risk)
    results["theta_tilde_norm"][idx].append(theta_tilde_norm)
    results["theta_hat_norm"][idx].append(theta_hat_norm)

    # selector = args.spiked_num * softmax(theta_hat / temperature)
    selector = repeated_argmax(theta_hat, args.spiked_num)
    y_hat = X.T @ selector
    theta_new = M.T @ y_hat
    theta_new_test_risk = mean_squared_error(
        X_test.T @ theta_new, y_tilde_test)
    theta_new_norm = np.linalg.norm(theta_new)

    results[f"theta_new_risk"][idx].append(theta_new_test_risk)
    results[f"theta_new_norm"][idx].append(theta_new_norm)

def experiment(args):
    # Initializes the results dictionary.
    results = {
        "risk_diff": [[] for _ in range(len(ds))],
        "theta_tilde_risk": [[] for _ in range(len(ds))],
        "theta_hat_risk": [[] for _ in range(len(ds))],
        "theta_new_risk": [[] for _ in range(len(ds))],
        "theta_tilde_norm": [[] for _ in range(len(ds))],
        "theta_hat_norm": [[] for _ in range(len(ds))],
        "theta_new_norm": [[] for _ in range(len(ds))],
    } 

    # Runs experiment trials.
    for idx, d in enumerate(ds):
        for _ in range(trials):
            experiment_trial(args, results, idx, d)
            
    # Computes the mean over all trials for each metric and value of d in the
    # results dictionary.
    for k, v in results.items():
        results[k] = np.mean(v, axis=1)

    return results
    
def main():
    for theta_star_type in theta_star_types:
        args.theta_star_type = theta_star_type
        for y_type in y_types:
            args.y_type = y_type
            for cov_type in cov_types:
                args.cov_type = cov_type

                # Runs experiments while iterating over hyperparameters.
                if cov_type == "isotropic":
                    print("Running isotropic experiments...")
                    results = [experiment(args)]
                elif cov_type == "spiked":
                    results = []
                    for spiked_size in spiked_sizes:
                        sub_results = []
                        args.spiked_size = spiked_size
                        for spiked_num in spiked_nums:
                            args.spiked_num = spiked_num
                            print((
                                "Running spiked experiments:"
                                f" size {spiked_size} num {spiked_num}..."
                            ))
                            sub_result = experiment(args)
                            sub_results.append(sub_result)
                        results.append(sub_results)
                elif cov_type == "poly":
                    results = []
                    for poly_exponent in poly_exponents:
                        print((
                            "Running poly experiments:"
                            f" exponent {poly_exponent}..."
                        ))
                        args.poly_exponent = poly_exponent
                        result = experiment(args)
                        results.append(result)

                # Plots the experiment results.
                plot(args, results)

if __name__ == "__main__":
    main()
