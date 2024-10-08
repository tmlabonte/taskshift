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

def set_display_metrics(args, fname):
    # Adds y_type to filename.
    if args.y_type == "gaussian":
        fname += "_gaussian"
        display_y_type = r"$y\sim\mathcal{N}(0,1)$"
    elif args.y_type == "sgn":
        fname += "_sgn"
        display_y_type = r"$y=sgn(\tilde{y})$"
    
    # Adds theta_star_type to filename.
    if args.theta_star_type == "k_sparse":
        vecs = "".join([str(j) for j in range(1, args.k_sparse_num + 1)])
        fname += f"_{args.k_sparse_num}-sparse"
        display_theta_star_type = fr"$\theta^\star={args.k_sparse_num}$-sparse"
    if args.theta_star_type == "step":
        fname += f"_step{args.step_val}"
        display_theta_star_type = rf"$\theta^\star=(1,{args.step_val},0,\dots)$"
    elif args.theta_star_type == "unif":
        fname += "_unif"
        display_theta_star_type = r"$\theta^\star\sim U(-1,1)$"

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
    n_range = np.arange(args.n_start, args.n_end + 1, args.n_step)

    # Defines a helper function for plotting the results.
    def plot_helper(
        metrics, display_metrics, fname, xmin=None, xmax=None, ymin=0, ymax=None
    ):
        fname, display_theta_star_type, display_y_type = \
                set_display_metrics(args, fname)

        start = (xmin - args.n_start) // args.n_step if xmin else 0
        end = (xmax - args.n_start) // args.n_step if xmax else len(n_range)

        # Plots specified range of data.
        for name, display_name in zip(metrics, display_metrics):
            if args.cov_type == "isotropic":
                label = display_name
            elif args.cov_type == "poly":
                label = display_name
                label += rf" poly=${args.poly_exponent}$"
            elif args.cov_type == "spiked":
                label = display_name
                spiked_ratio = f"n^{{{args.spiked_ratio_pow}}}/d"
                label += rf" ratio=${spiked_ratio}$"
                label += rf" num=${args.spiked_num}$"

            plt.plot(
                n_range[start:end],
                results[name][start:end],
                label=label,
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

        plt.ylim(ymin, ymax)
        plt.legend()
        plt.title(title)
        plt.xlabel("n")
        plt.savefig(fname, bbox_inches="tight", dpi=600)
        plt.clf()

    # Plots the results.
    plot_helper(
        ["independent", "correlated"],#, "anticorrelated"],
        ["independent", "correlated"],#, "anticorrelated"],
        "ind",
    )

def experiment_trial(args, results, idx, n):
    d = math.ceil(args.d_coef * n ** args.d_pow)

    # Generates the ground-truth regressor.
    if args.theta_star_type == "k_sparse":
        theta_star = torch.zeros(d)
        theta_star[:args.k_sparse_num] = 1
    elif args.theta_star_type == "step":
        theta_star = torch.zeros(d)
        theta_star[0] = 1
        theta_star[1] = args.step_val
    elif args.theta_star_type == "unif":
        theta_star = -2 * torch.random(d) + 1 # theta_star ~ Unif(-1, 1).
        theta_star = theta_star / torch.linalg.vector_norm(theta_star, ord=2)
    args.var = torch.linalg.vector_norm(theta_star, ord=2).item()

    # Generates the covariance matrix.
    if args.cov_type == "isotropic":
        # Generates an isotropic (identity) covariance matrix.
        cov_diag = torch.ones(d)
    elif args.cov_type == "poly":
        # Generates a polynomial decay covariance matrix which is diagonal
        # with the jth entry being j^{-poly_exponent}.
        cov_diag = torch.tensor([
            j ** -args.poly_exponent for j in range(1, d + 1)
        ])
    elif args.cov_type == "spiked":
        # Generates a spiked covariance matrix which is diagonal with 1 in the
        # first spiked_num entries and n^alpha / d in the rest.
        cov_diag = torch.ones(d)
        for j in range(args.spiked_num, d):
            cov_diag[j] = n ** args.spiked_ratio_pow / d

    # Generates the train data and labels using the ground-truth regressor.
    # The distribution D is equivalent to a MultivariateNormal but uses a
    # trick to save memory when covariance is diagonal.
    D = Independent(Normal(0, cov_diag.sqrt()), 1)
    X = D.sample((n,)) # n x d
    y_tilde = X @ theta_star # n

    # Generates classifier data as either the signs of the regression labels
    # or as independent standard Gaussians.
    if args.y_type == "sgn":
        y = torch.sign(y_tilde) # n
    elif args.y_type == "gaussian":
        y = torch.normal(0, 1, (n,)) # n

    # Computes quantities of interest.
    M = torch.cholesky_inverse(torch.linalg.cholesky(X @ X.T)) # n x n
    C = M @ X # n x d
    C = C * cov_diag[None, :] # multiply column c_j by element diag(j). n x d
    C = C @ X.T @ M # n x n

    normal = torch.normal(0, 1, (n,))
    independent = normal @ C @ normal
    correlated = (y - y_tilde) @ C @ (y - y_tilde)
    #independent = torch.norm(normal)
    #correlated = torch.norm((y-y_tilde))
    #anticorrelated = (-y) @ C @ (-y)
    anticorrelated = 0.

    # Adds relevant metrics to the results dictionary.
    results["independent"][idx].append(independent)
    results["correlated"][idx].append(correlated)
    results["anticorrelated"][idx].append(anticorrelated)

def experiment(args):
    n_range = list(range(args.n_start, args.n_end + 1, args.n_step))

    # Initializes the results dictionary.
    results = {
        "independent": [[] for _ in range(len(n_range))],
        "correlated": [[] for _ in range(len(n_range))],
        "anticorrelated": [[] for _ in range(len(n_range))],
    } 

    # Runs experiment trials.
    for idx, n in enumerate(n_range):
        for t in range(args.trials):
            if t == 0:
                print(f"Running n={n}...")
            experiment_trial(args, results, idx, n)
            
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
    parser.add("--d_coef", default=1, type=float)
    parser.add("--d_pow", default=2, type=float)
    parser.add("--k_sparse_num", default=1, type=int)
    parser.add("--n_start", default=50, type=int)
    parser.add("--n_step", default=100, type=int)
    parser.add("--n_end", default=1050, type=int)
    parser.add("--out_dir", default="out")
    parser.add("--poly_exponent", default=2, type=float)
    parser.add("--spiked_num", default=1, type=int)
    parser.add("--spiked_ratio_pow", default=0.66, type=float)
    parser.add("--step_val", default=0.5, type=float)
    parser.add("--theta_star_type", choices=["k_sparse", "step", "unif"], default="k_sparse")
    parser.add("--trials", default=5, type=int)
    parser.add("--y_type", choices=["gaussian", "sgn"], default="sgn")
    args = parser.parse_args()

    # Spiked num should align with k_sparse_num or step_num.
    if args.cov_type == "spiked":
        if args.theta_star_type == "k_sparse" and \
                args.k_sparse_num != args.spiked_num:
            warnings.warn((
                "Spiked covariance and k-sparse model are specified"
                " but k_sparse_num != spiked_num."
            ))
        elif args.theta_star_type == "step" and args.spiked_num != 2:
            warnings.warn("Step model is specified but spiked_num != 2.")

    main(args)

