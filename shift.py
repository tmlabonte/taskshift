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
    return torch.zeros(len(x)).scatter_(0, torch.topk(x, k)[1], 1)

def set_display_metrics(args, fname):
    # Adds y_type to filename.
    if args.y_type == "gaussian":
        fname += "_gaussian"
        display_y_type = r"$y\sim\mathcal{N}(0,1)$"
    elif args.y_type == "sgn":
        fname += "_sgn"
        display_y_type = r"$y=sgn(\tilde{y})$"
    
    # Adds theta_star_type to filename.
    if args.theta_star_type == "sparse":
        vecs = "".join([str(j) for j in range(1, args.sparse_num + 1)])
        fname += f"_{args.sparse_num}-sparse"
        display_theta_star_type = fr"$\theta^\star={args.sparse_num}$-sparse"
    elif args.theta_star_type == "gaussian":
        fname += "_gaussian"
        display_theta_star_type = r"$\theta^\star\sim N(0,1)$"

    # Adds cov_type to filename.
    if args.cov_type == "isotropic":
        fname += "_isotropic.png"
    elif args.cov_type == "poly":
        fname += "_poly.png"
    elif args.cov_type == "spiked":
        fname += "_spiked.png"

    fname = os.path.join(args.out_dir, fname)

    return fname, display_theta_star_type, display_y_type

def plot(args, results):
    if args.x_axis_type == "n":
        x_range = list(range(args.n_start, args.n_end + 1, args.n_step))
    elif args.x_axis_type == "d":
        x_range = list(range(args.d_start, args.d_end + 1, args.d_step))

    # Defines a helper function for plotting the results.
    def plot_helper(
        metrics, display_metrics, fname, xmin=None, xmax=None, ymin=0, ymax=None
    ):
        fname, display_theta_star_type, display_y_type = \
                set_display_metrics(args, fname)

        if args.x_axis_type == "n":
            start = (xmin - args.n_start) // args.n_step if xmin else 0
            end = (xmax - args.n_start) // args.n_step if xmax else len(x_range)
        elif args.x_axis_type == "d":
            start = (xmin - args.d_start) // args.d_step if xmin else 0
            end = (xmax - args.d_start) // args.d_step if xmax else len(x_range)

        # Plots specified range of data.
        for name, display_name in zip(metrics, display_metrics):
            if args.cov_type == "isotropic":
                label = display_name
            elif args.cov_type == "poly":
                label = display_name
            elif args.cov_type == "spiked":
                label = display_name

            plt.plot(
                x_range[start:end],
                results[name][start:end],
                label=label,
                linewidth=5,
            )

        # Automatically crops the y-axis if a bound on the x-axis is specified.
        if (xmin or xmax) and not ymax:
            ymax = max(torch.tensor([
                result[metric][start:end] for metric in metrics
            ]).flatten())

        title = fr"{display_theta_star_type}, {display_y_type}, "
        title += fr"$\Sigma=${args.cov_type}, $d={args.d_coef}n^{{{args.d_pow}}}$, "
        title += fr"$\|\theta^\star\|_2^2={round(args.var, 3)}$, "
        title += f"{args.trials} trials"
        xlabel = "$n$" if args.x_axis_type == "n" else "$d$"

        plt.ylim(ymin, ymax)
        plt.legend()
        # plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Risk")
        plt.grid(alpha=0.5)
        plt.savefig(fname, bbox_inches="tight", dpi=600)
        plt.clf()

    if args.x_axis_type == "n":
        if args.y_type == "sgn":
            if args.cov_type == "isotropic":
                ymax = 2 * args.var
            elif args.cov_type == "poly":
                ymax = 1.5
            elif args.cov_type == "spiked":
                ymax = None
        elif args.y_type == "gaussian":
            ymax = 2 * args.var
    elif args.x_axis_type == "d":
        ymax = 1.5

    # Plots the results.
    plot_helper(
        ["risk_diff"],
        [r"$R(\hat{\theta})-R(\theta)$"],
        "diff",
        ymax=ymax,
    )
    plot_helper(
        ["theta_tilde_risk", "theta_hat_risk", "theta_new_risk"],
        [r"$R(\theta)$", r"$R(\hat{\theta})$", r"$R(\hat{\theta}^\prime)$"],
        "risk",
        ymax=ymax,
    )
    plot_helper(
        ["theta_tilde_norm", "theta_hat_norm", "theta_new_norm"],
        [r"$\|\theta\|_2$", r"$\|\hat{\theta}\|_2$", r"$\|\hat{\theta}^\prime\|_2$"],
        "norm",
        ymax=2 * args.var,
    )

def make_selector(args, theta_hat, theta_star):
    return selector

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

def experiment_trial(args, results, idx, x):
    if args.x_axis_type == "n":
        n = x
        d = math.ceil(args.d_coef * n ** args.d_pow)
    elif args.x_axis_type == "d":
        d = x
        n = math.floor((d  / args.d_coef) ** (1 / args.d_pow))

    # Generates the ground-truth regressor.
    if args.theta_star_type == "sparse":
        theta_star = torch.zeros(d)
        theta_star[:args.sparse_num] = 1
    elif args.theta_star_type == "gaussian":
        theta_star = torch.normal(0, 1, size=(d,))
        theta_star = theta_star / torch.linalg.vector_norm(theta_star, ord=2)

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
        # Generates a spiked covariance matrix which is diagonal with 1 in the
        # first spiked_num entries and n^alpha / d in the rest.
        cov_diag = torch.ones(d)
        for j in range(args.spiked_num, d):
            cov_diag[j] = n ** args.spiked_ratio_pow / d

    # Computes total variance/signal strength || cov^{1/2} theta* ||_2^2.
    var_vec = torch.sqrt(cov_diag) * theta_star
    args.var = torch.linalg.vector_norm(var_vec).item() ** 2

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

        theta_tilde_norm = torch.linalg.vector_norm(theta_tilde, ord=2)
        theta_hat_norm = torch.linalg.vector_norm(theta_hat, ord=2)
        risk_diff = (theta_hat_test_risk - theta_tilde_test_risk)

        # Adds relevant metrics to the results dictionary.
        results["risk_diff"][idx].append(risk_diff)
        results["theta_tilde_risk"][idx].append(theta_tilde_test_risk)
        results["theta_hat_risk"][idx].append(theta_hat_test_risk)
        results["theta_tilde_norm"][idx].append(theta_tilde_norm)
        results["theta_hat_norm"][idx].append(theta_hat_norm)

        # Computes new predictor using theta_hat by taking the top-k indices.
        if args.theta_star_type == "sparse":
            selector = topk_inds(theta_hat, args.sparse_num)

    y_hat = X @ selector # n
    if args.solver == "direct":
        theta_new = M @ y_hat # d
    elif args.solver == "gd":
        theta_new = gradient_descent(X, y_hat)
    theta_new_test_risk = F.mse_loss(
        X_test @ theta_new, y_tilde_test)
    theta_new_norm = torch.linalg.vector_norm(theta_new, ord=2)

    results[f"theta_new_risk"][idx].append(theta_new_test_risk)
    results[f"theta_new_norm"][idx].append(theta_new_norm)

def experiment(args):
    if args.x_axis_type == "n":
        x_range = list(range(args.n_start, args.n_end + 1, args.n_step))
    elif args.x_axis_type == "d":
        x_range = list(range(args.d_start, args.d_end + 1, args.d_step))

    # Initializes the results dictionary.
    results = {
        "risk_diff":        [[] for _ in range(len(x_range))],
        "theta_tilde_risk": [[] for _ in range(len(x_range))],
        "theta_hat_risk":   [[] for _ in range(len(x_range))],
        "theta_new_risk":   [[] for _ in range(len(x_range))],
        "theta_tilde_norm": [[] for _ in range(len(x_range))],
        "theta_hat_norm":   [[] for _ in range(len(x_range))],
        "theta_new_norm":   [[] for _ in range(len(x_range))],
    } 

    # Runs experiment trials.
    for idx, x in enumerate(x_range):
        for t in range(args.trials):
            if t == 0 and x % 100 == 0:
                print(f"Running {args.x_axis_type}={x}...")
            experiment_trial(args, results, idx, x)
            
    # Computes the mean over all trials for each metric and value of d in the
    # results dictionary.
    for metric, value in results.items():
        results[metric] = torch.mean(torch.tensor(value), axis=1).cpu().numpy()

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
    parser.add("--sparse_inds", default=[], nargs="*", type=int)
    parser.add("--sparse_vals", default=[], nargs="*", type=float)
    parser.add("--spiked_p", default=2, type=float)
    parser.add("--spiked_q", default=0.5, type=float)
    parser.add("--spiked_r", default=0, type=float)
    parser.add("--theta_star_type", choices=["sparse", "gaussian"], default="sparse")
    parser.add("--trials", default=5, type=int)
    # Either scales n and d together ("d") or holds n constant ("n").
    parser.add("--x_axis_type", choices=["n", "d"], default="n")
    parser.add("--y_type", choices=["gaussian", "sgn"], default="sgn")
    args = parser.parse_args()

    if args.spiked_p <= 1:
        raise ValueError(f"Found p = {args.spiked_p} but requires p > 1.")
    if args.spiked_q <= 0 or args.spiked_q > p - r:
        raise ValueError(f"Found q = {args.spiked_q} but requires 0 < q < p - r.")
    if args.spiked_r < 0 or args.spiked_r >= 1:
        raise ValueError(f"Found r = {args.spiked_r} but requires 0 <= r < 1.")

    main(args)

