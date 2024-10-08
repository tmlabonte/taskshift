from configargparse import Parser
from distutils.util import strtobool
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import torch
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
import torch.nn.functional as F
import warnings

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 18})

def topk_inds(x, k):
    """Returns mask of top-k elements of x."""
    return torch.zeros(len(x)).scatter_(0, torch.topk(x, k)[1], 1).bool()

def postprocess(theta_hat, X, empirical_support, true_variance):
    """Scales classification MNI using knowledge of variance."""

    # Computes empirical variances for the support.
    # empirical_variances = torch.var(X, dim=0)[empirical_support]

    # Computes postprocessed predictor using formula and variances.
    sgn = torch.sign(theta_hat[empirical_support])
    theta_new = theta_hat[empirical_support]
    theta_new *= np.sqrt((np.pi / 2) * true_variance)
    theta_new = sgn * theta_new

    theta_final = torch.zeros(len(theta_hat))
    theta_final[empirical_support] = theta_new

    return theta_final

def set_fname(args, fname):
    # Adds y_type to filename.
    if args.y_type == "gaussian":
        fname += "_gaussian"
    elif args.y_type == "sgn":
        fname += "_sgn"
    
    # Adds theta_star_type to filename.
    if args.theta_star_type == "sparse":
        fname += "_sparse"
    elif args.theta_star_type == "gaussian":
        fname += "_gaussian"

    # Adds cov_type to filename.
    if args.cov_type == "isotropic":
        fname += "_isotropic.png"
    elif args.cov_type == "poly":
        fname += "_poly.png"
    elif args.cov_type == "spiked":
        fname += "_spiked.png"

    fname = os.path.join(args.out_dir, fname)

    return fname

def plot(args, results):
    n_range = list(range(args.n_start, args.n_end + 1, args.n_step))

    # Defines a helper function for plotting the results.
    def plot_helper(
        metrics, display_metrics, fname, ymin=0, ymax=None
    ):
        fname = set_fname(args, fname)
        
        # Plots specified range of data.
        for metric, display_metric in zip(metrics, display_metrics):
            plt.plot(
                n_range,
                metric,
                label=display_metric,
                linewidth=5,
            )

        plt.ylim(ymin, ymax)
        plt.legend()
        plt.xlabel(r"$n$")
        if "risk" in fname:
            plt.ylabel("Risk")
        plt.grid(alpha=0.5)
        plt.savefig(fname, bbox_inches="tight", dpi=600)
        plt.clf()

    # Puts back missing dimension when len(args.sparse_inds) = 1.
    try:
        len_support = len(results["theta_hat_support"][0])
    except TypeError:
        results["theta_hat_support"] = results["theta_hat_support"][..., np.newaxis]
        len_support = len(results["theta_hat_support"][0])

    # Plots results.
    plot_helper(
        [results["theta_hat_support"][:, s] for s in range(len_support)] + [results["theta_hat_other_avg"]],
        [r"$\hat{k}_{j}$".format(k=r"\theta", j=j) for j in range(1, len_support + 1)] +
            ["Average of non-support"],
        "support",
    )
    plot_helper(
        [results["theta_new_diff"]],
        [r"$\|\hat{\theta}^\prime - \theta^\star\|_2$"],
        "diff",
    )
    plot_helper(
        [results["theta_tilde_risk"], results["theta_hat_risk"], results["theta_new_risk"]],
        [r"$L(\tilde{\theta})$", r"$L(\hat{\theta})$", r"$L(\hat{\theta}^\prime)$"],
        "risk",
    )
    plot_helper(
        [results["theta_star_1"], results["theta_tilde_1"], results["theta_hat_1"], results["theta_new_1"]],
        [r"$\theta^\star_1$", r"$\tilde{\theta}_1$", r"$\hat{\theta}_1$", r"$\hat{\theta}^\prime_1$"],
        "vals",
    )
    
def gradient_descent(X, y):
    max_steps = 1000
    early_stop_loss = 0.001
    learning_rate = 0.05

    theta = torch.zeros(X.shape[1], requires_grad=True)
    for _ in range(max_steps):
        loss = F.mse_loss(X @ theta, y)
        if loss.item() < early_stop_loss:
            return theta
        elif loss.item() > 10:
            raise ValueError("Exploding gradient detected")

        loss.backward()
        with torch.no_grad():
            theta -= learning_rate * theta.grad.data
            theta.grad.data.zero_()

    return theta

def experiment_trial(args, results, idx, n):
    # Sets dimension d and spiked parameters.
    if args.cov_type == "spiked":
        d = int(n ** args.spiked_p)
        s = int(n ** args.spiked_r)
        a = n ** -args.spiked_q
    else:
        d = int(args.d_coef * n ** args.d_pow)
    
    # Generates the covariance matrix.
    if args.cov_type == "isotropic":
        # Generates an isotropic (identity) covariance matrix.
        cov_diag = torch.ones(d)
    elif args.cov_type == "poly":
        # Generates a polynomial decay covariance matrix which is diagonal
        # with the jth entry being j^{-poly_pow}.
        cov_diag = torch.tensor([
            j ** -args.poly_pow for j in range(1, d + 1)
        ])
    elif args.cov_type == "spiked":
        # Generates a spiked covariance matrix which is diagonal with ad/s in
        # the first s entries and (1-a)d/(d-s) otherwise.
        cov_diag = torch.ones(d)
        for j in range(s):
            cov_diag[j] = (a * d) / s
        for j in range(s, d):
            cov_diag[j] = ((1 - a) * d) / (d - s)

    # Generates the ground-truth regressor.
    if args.theta_star_type == "sparse":
        theta_star = torch.zeros(d)
        cov_on_support = cov_diag[torch.tensor(args.sparse_inds)]
        theta_star[torch.tensor(args.sparse_inds)] = torch.tensor(args.sparse_vals) / torch.sqrt(cov_on_support)
    elif args.theta_star_type == "gaussian":
        theta_star = torch.normal(0, 1, size=(d,))
        theta_star = theta_star / torch.linalg.vector_norm(theta_star, ord=2)

    # Computes total variance/signal strength.
    variance_vector = torch.sqrt(cov_diag) * theta_star
    true_variance = torch.linalg.vector_norm(variance_vector).item() ** 2

    # Generates the train data and labels using the ground-truth regressor.
    # The distribution D is equivalent to a MultivariateNormal but uses a
    # trick to save memory when covariance is diagonal.
    D = Independent(Normal(0, cov_diag.sqrt()), 1)
    X = D.sample((n,)) # n x d
    y_tilde = X @ theta_star # n

    # Generates the test data and labels using the ground-truth regressor.
    X_test = D.sample((args.n_test,)) # n_test x d
    y_tilde_test = X_test @ theta_star # n_test

    # Generates classifier data as either the signs of the regression labels
    # or as independent standard Gaussians.
    if args.y_type == "sgn":
        y = torch.sign(y_tilde) # n
    elif args.y_type == "gaussian":
        y = torch.normal(0, 1, (n,)) # n

    # Computes the minimum-norm interpolators for regression and classification.
    if args.solver == "direct":
        M = X.T @ torch.cholesky_inverse(torch.linalg.cholesky(X @ X.T)) # d x n
        theta_tilde = M @ y_tilde # d
        theta_hat = M @ y # d
    elif args.solver == "gd":
        theta_tilde = gradient_descent(X, y_tilde)
        theta_hat = gradient_descent(X, y)

    with torch.no_grad(): # IMPORTANT
        # Computes the test risk of the minimum-norm interpolators.
        theta_tilde_test_risk = F.mse_loss(
            X_test @ theta_tilde, y_tilde_test)
        theta_hat_test_risk = F.mse_loss(
            X_test @ theta_hat, y_tilde_test)

        # Computes other metrics of interest.
        true_support = torch.zeros(len(theta_hat), dtype=bool)
        true_support[torch.tensor(args.sparse_inds)] = True
        theta_hat_support = theta_hat[true_support]
        theta_hat_other_avg = torch.mean(theta_hat[~true_support])

        # Adds metrics to the results dictionary.
        results["theta_hat_1"][idx].append(theta_hat[0])
        results["theta_hat_risk"][idx].append(theta_hat_test_risk)
        results["theta_hat_support"][idx].append(theta_hat_support)
        results["theta_hat_other_avg"][idx].append(theta_hat_other_avg)
        results["theta_star_1"][idx].append(theta_star[0])
        results["theta_tilde_1"][idx].append(theta_tilde[0])
        results["theta_tilde_risk"][idx].append(theta_tilde_test_risk)

    # Computes new predictor using postprocessing algorithm.
    if args.theta_star_type == "sparse":
        # Selects top-k indices of theta_hat (assumes we know k).
        empirical_support = topk_inds(theta_hat, len(args.sparse_inds))

        # TODO: Implement few-shot least-squares postprocessing.

        # Scales the classification MNI using knowledge of variance.
        theta_new = postprocess(theta_hat, X, empirical_support, true_variance)

        with torch.no_grad(): # IMPORTANT
            # Computes the test risk of the postprocessed predictor.
            theta_new_test_risk = F.mse_loss(
                X_test @ theta_new, y_tilde_test)

            # Computes other metrics of interest.
            theta_new_diff = torch.linalg.vector_norm(theta_star - theta_new)
            
            # Adds metrics to the results dictionary.
            results["theta_new_1"][idx].append(theta_new[0])
            results["theta_new_risk"][idx].append(theta_new_test_risk)
            results["theta_new_diff"][idx].append(theta_new_diff)

def experiment(args):
    n_range = list(range(args.n_start, args.n_end + 1, args.n_step))

    # Initializes the results dictionary.
    results = {
        "theta_hat_1":         [[] for _ in range(len(n_range))],
        "theta_hat_risk":      [[] for _ in range(len(n_range))],
        "theta_hat_support":   [[] for _ in range(len(n_range))],
        "theta_hat_other_avg": [[] for _ in range(len(n_range))],
        "theta_new_1"  :       [[] for _ in range(len(n_range))],
        "theta_new_risk":      [[] for _ in range(len(n_range))],
        "theta_new_diff":      [[] for _ in range(len(n_range))],
        "theta_star_1":       [[] for _ in range(len(n_range))],
        "theta_tilde_1":       [[] for _ in range(len(n_range))],
        "theta_tilde_risk":    [[] for _ in range(len(n_range))],
    } 

    # Runs experiment trials.
    for idx, n in enumerate(n_range):
        for t in range(args.trials):
            if t == 0 and n % 100 == 0:
                print(f"Running n={n}...")
            experiment_trial(args, results, idx, n)
            
    # Computes the mean over all trials for each metric and value of n in the
    # results dictionary.
    for metric, value in results.items():
        try:
            value = torch.tensor(value)
        except ValueError:
            value = torch.stack([torch.stack(v) for v in value])
        results[metric] = torch.mean(value, dim=1).cpu().numpy()

    return results

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    results = experiment(args)
    plot(args, results)

if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=False,
    )

    # Loads configuration parameters into parser.
    parser.add("--cov_type", choices=["isotropic", "poly", "spiked"], default="spiked")
    parser.add("--cuda", default=True, type=lambda x: bool(strtobool(x)))
    parser.add("--d_coef", default=10, type=float)
    parser.add("--d_pow", default=1, type=float)
    parser.add("--d_start", default=5, type=int)
    parser.add("--d_step", default=5, type=int)
    parser.add("--d_end", default=1005, type=int)
    parser.add("--n", default=100, type=int)
    parser.add("--n_start", default=50, type=int)
    parser.add("--n_step", default=50, type=int)
    parser.add("--n_end", default=2500, type=int)
    parser.add("--n_test", default=100, type=int)
    parser.add("--out_dir", default="out")
    parser.add("--poly_pow", default=2, type=float)
    parser.add("--solver", choices=["direct", "gd"], default="direct")
    parser.add("--sparse_inds", default=[0], nargs="*", type=int)
    parser.add("--sparse_vals", default=[1.], nargs="*", type=float)
    parser.add("--spiked_p", default=1.5, type=float)
    parser.add("--spiked_q", default=0.5, type=float)
    parser.add("--spiked_r", default=0, type=float)
    parser.add("--theta_star_type", choices=["gaussian", "sparse"], default="sparse")
    parser.add("--trials", default=5, type=int)
    parser.add("--y_type", choices=["gaussian", "sgn"], default="sgn")
    args = parser.parse_args()

    if args.spiked_p <= 1:
        raise ValueError(f"Found p = {args.spiked_p} but requires p > 1.")
    if args.spiked_q <= 0 or args.spiked_q > args.spiked_p - args.spiked_r:
        raise ValueError(f"Found q = {args.spiked_q} but requires 0 < q < p - r.")
    if args.spiked_r < 0 or args.spiked_r >= 1:
        raise ValueError(f"Found r = {args.spiked_r} but requires 0 <= r < 1.")

    main(args)

