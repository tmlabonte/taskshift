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
    """Returns mask of top-k elements of abs(x)."""
    return torch.zeros(len(x)).scatter_(0, torch.topk(torch.abs(x), k)[1], 1).bool()

def scaling(theta_hat, empirical_support, true_variance):
    """Scales classification MNI using knowledge of variance."""

    # Computes postprocessed predictor using formula.
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
    n_postprocess_range = list(range(
        args.n_postprocess_start,
        args.n_postprocess_end + 1,
        args.n_step,
    ))

    # Defines a helper function for plotting the results.
    def plot_helper(
        x, metrics, display_metrics, fname, colors=None,
    ):
        fname = set_fname(args, fname)
        
        # Plots specified range of data.
        ma = 0
        mi = 0.005
        for j, (metric, display_metric) in enumerate(zip(metrics, display_metrics)):
            if max(results[metric]) > ma:
                ma = max(results[metric])
            if min(results[metric]) < mi:
                mi = min(results[metric])

            color=None
            if colors:
                color = colors[j]

            plt.plot(
                x,
                results[metric],
                label=display_metric,
                linewidth=5,
                color=color,
            )

        if "support" in fname:
            plt.xlabel("Index")
        elif "risk" in fname:
            plt.ylabel("Risk")
            plt.xlabel(r"$n$")
        elif "postprocess" in fname:
            plt.ylabel("Risk")
            plt.xlabel(r"$m$")

        if args.cov_type == "spiked" and not args.outside_support:
            plt.ylim(min(0, mi - 0.005), min(1, ma + 0.005))
        plt.legend()
        plt.grid(alpha=0.5)
        plt.savefig(fname, bbox_inches="tight", dpi=600)
        plt.clf()

    # Plots results.
    plot_helper(
        np.arange(100),
        ["theta_hat_begin", "theta_hat_end"],
        [rf"$n={args.n_end // 10}$", rf"$n={args.n_end}$"],
        "support",
    )
    plot_helper(
        n_range,
        ["theta_tilde_risk", "theta_hat_risk", "theta_new_risk"],
        [r"$L(\tilde{\theta})$", r"$L(\hat{\theta})$", r"$L(\hat{\theta}^\prime)$"],
        "risk",
        colors=["C1", "C2", "C0"],
    )
    plot_helper(
        n_postprocess_range,
        ["theta_new_risks"],
        [r"$L(\hat{\theta}^\prime)$"],
        "postprocess",
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
        theta_star[torch.tensor(args.sparse_inds) - 1] = torch.tensor(args.sparse_vals) / torch.sqrt(cov_on_support)

        if args.outside_support:
            max_s = int(args.n_end ** args.spiked_r)
            theta_star[max_s + 1] = args.outside_support_val / torch.sqrt(cov_diag[max_s + 1])
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
    y_test = X_test @ theta_star # n_test

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
            X_test @ theta_tilde, y_test)
        theta_hat_test_risk = F.mse_loss(
            X_test @ theta_hat, y_test)

        # Adds metrics to the results dictionary.
        results["theta_hat_risk"][idx].append(theta_hat_test_risk)
        results["theta_tilde_risk"][idx].append(theta_tilde_test_risk)

        # TODO: This part is hacky; depending on n_step it may not be // 10.
        if n == args.n_end // 10:
            results["theta_hat_begin"].append(theta_hat[:100])
        elif n == args.n_end:
            results["theta_hat_end"].append(theta_hat[:100])

    # Computes new predictor using postprocessing algorithm.
    if args.theta_star_type == "sparse":
        # Selects top-k indices of theta_hat (assumes we know k).
        len_sparse = len(args.sparse_inds)
        if args.outside_support:
            len_sparse += 1
        empirical_support = topk_inds(theta_hat, len_sparse)

        # Postprocesses either using least squares on few-shot regression data
        # or scaling the classification MNI using knowledge of variance.
        if args.postprocess == "least_squares":
            n_postprocess_range = range(
                args.n_postprocess_start,
                args.n_postprocess_end + 1,
                args.n_postprocess_step,
            )
            theta_new_test_risks = []
            for n_postprocess in n_postprocess_range:
                X_postprocess = D.sample((n_postprocess,)) # n_postprocess x d

                # Performs dimension reduction using knowledge of the support.
                X_postprocess = X_postprocess[:, empirical_support] # n_postprocess x k
                y_postprocess = X_postprocess @ theta_star[empirical_support] # n_postprocess

                # Adds Gaussian noise to regression labels.
                noise = torch.normal(0, 1, (len(y_postprocess),))
                y_postprocess += noise

                # Computes least-squares estimator for few-shot regression data.
                if args.solver == "direct":
                    A = torch.linalg.cholesky(X_postprocess.T @ X_postprocess) # k x k
                    M = torch.cholesky_inverse(A) @ X_postprocess.T # k x n_postprocess
                    theta_new = M @ y_postprocess # k
                elif args.solver == "gd":
                    theta_new = gradient_descent(X_postprocess, y_postprocess)

                with torch.no_grad(): # IMPORTANT
                    # Computes the test risk of the postprocessed predictor.
                    theta_new_test_risk = F.mse_loss(
                        X_test[:, empirical_support] @ theta_new, y_test)

                    # Adds metrics to the results dictionary.
                    theta_new_test_risks.append(theta_new_test_risk)
            results["theta_new_risks"][idx].append(theta_new_test_risks)

        elif args.postprocess == "scaling":
            theta_new = scaling(theta_hat, empirical_support, true_variance)

            with torch.no_grad(): # IMPORTANT
                # Computes the test risk of the postprocessed predictor.
                theta_new_test_risk = F.mse_loss(
                    X_test @ theta_new, y_test)

                # Adds metrics to the results dictionary.
                results["theta_new_risk"][idx].append(theta_new_test_risk)

def experiment(args):
    n_range = list(range(args.n_start, args.n_end + 1, args.n_step))

    # Initializes the results dictionary.
    results = {
        "theta_hat_risk":      [[] for _ in range(len(n_range))],
        "theta_hat_begin":     [],
        "theta_hat_end":       [],
        "theta_new_risks":     [[] for _ in range(len(n_range))],
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
        except:
            value = torch.stack(value).T
        results[metric] = torch.mean(value, dim=1).cpu().numpy()

    results["theta_new_risk"] = results["theta_new_risks"][:, -1] # last n_postprocess
    results["theta_new_risks"] = results["theta_new_risks"][-1, :] # last n

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
    parser.add("--d_coef", default=1, type=float)
    parser.add("--d_pow", default=1.5, type=float)
    parser.add("--n_postprocess_start", default=100, type=int)
    parser.add("--n_postprocess_step", default=50, type=int)
    parser.add("--n_postprocess_end", default=1000, type=int)
    parser.add("--n_start", default=50, type=int)
    parser.add("--n_step", default=50, type=int)
    parser.add("--n_end", default=2500, type=int)
    parser.add("--n_test", default=100, type=int)
    parser.add("--out_dir", default="out")
    parser.add("--outside_support", default=False, type=lambda x: bool(strtobool(x)))
    parser.add("--outside_support_val", default=1, type=float)
    parser.add("--poly_pow", default=2, type=float)
    parser.add("--postprocess", choices=["least_squares", "scaling"], default="least_squares")
    parser.add("--solver", choices=["direct", "gd"], default="direct")
    parser.add("--sparse_inds", default=[0], nargs="*", type=int)
    parser.add("--sparse_vals", default=[1.], nargs="*", type=float)
    parser.add("--spiked_p", default=1.5, type=float)
    parser.add("--spiked_q", default=0.5, type=float)
    parser.add("--spiked_r", default=0.25, type=float)
    parser.add("--theta_star_type", choices=["gaussian", "sparse"], default="sparse")
    parser.add("--trials", default=10, type=int)
    parser.add("--y_type", choices=["gaussian", "sgn"], default="sgn")
    args = parser.parse_args()

    if args.spiked_p <= 1:
        raise ValueError(f"Found p = {args.spiked_p} but requires p > 1.")
    if args.spiked_q <= 0 or args.spiked_q > args.spiked_p - args.spiked_r:
        raise ValueError(f"Found q = {args.spiked_q} but requires 0 < q < p - r.")
    if args.spiked_r < 0 or args.spiked_r >= 1:
        raise ValueError(f"Found r = {args.spiked_r} but requires 0 <= r < 1.")

    main(args)

