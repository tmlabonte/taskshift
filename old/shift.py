from configargparse import Parser
from distutils.util import strtobool
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import warnings

plt.rcParams["text.usetex"] = True

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
    n_range = torch.arange(args.n_start, args.n_end + 1, args.n_step)
    d_range = n_range ** args.d_exponent

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
                label += rf" ratio=${args.spiked_ratio}$"
                label += rf" num=${args.spiked_num}$"

            if "theta_new" in name:
                label += rf" temp=${args.temperature}$"

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
        title += fr"$\Sigma=${args.cov_type}, $d=n^{args.d_exponent}$, "
        title += fr"$\|\theta^\star\|_2^2={round(args.var, 3)}$, "
        title += f"{args.trials} trials"

        plt.ylim(ymin, ymax)
        plt.legend()
        plt.title(title)
        plt.xlabel("n")
        plt.savefig(fname, bbox_inches="tight", dpi=600)
        plt.clf()

    if args.y_type == "sgn":
        if args.cov_type == "isotropic":
            ymax = args.var
        elif args.cov_type == "poly":
            ymax = 1.5
        elif args.cov_type == "spiked":
            ymax = args.var + 0.25
    elif args.y_type == "gaussian":
        ymax = 2 * args.var

    # Plots the results.
    plot_helper(
        ["risk_diff"],
        [r"$L(\hat{\theta})-L(\tilde{\theta})$"],
        "diff",
        ymax=ymax,
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
        ymax=2 * args.var,
    )

def make_selector(args, theta_hat, theta_star):
    if args.theta_star_type == "k-sparse":
        k = args.k_sparse_num
    elif args.cov_type == "spiked":
        k = args.spiked_num
    elif args.theta_star_type == "step":
        k = 2
    else:
        raise NotImplementedError()

    sgn_mask = torch.sign(theta_hat)
    inds = topk_inds(torch.abs(theta_hat), k).bool()
    softmaxed_topk = F.softmax(
        torch.abs(theta_hat)[inds] / args.temperature, dim=0)

    selector = torch.zeros(len(theta_hat))
    selector[inds] = softmaxed_topk
    selector *= torch.linalg.vector_norm(theta_star, ord=1)
    selector *= sgn_mask

    return selector

def experiment_trial(args, results, idx, n, d):
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
        cov = torch.eye(d)
    elif args.cov_type == "poly":
        # Generates a polynomial decay covariance matrix which is diagonal
        # with the jth entry being j^{-poly_exponent}.
        eigenvalues = torch.tensor([
            j ** -args.poly_exponent for j in range(1, d + 1)
        ])
        cov = torch.diag(eigenvalues)
    elif args.cov_type == "spiked":
        # Generates a spiked covariance matrix which is diagonal with 1 in the
        # first spiked_num entries and n / d in the rest.
        cov = torch.eye(d)
        for j in range(min(args.spiked_num, d)):
            cov[j, j] = n / d

    # Generates the train data and labels using the ground-truth regressor.
    D = MultivariateNormal(torch.zeros(d), cov)
    X = D.sample((n,)) # n x d
    y_tilde = X @ theta_star # n

    # Generates classifier data as either the signs of the regression labels
    # or as independent standard Gaussians.
    if args.y_type == "sgn":
        y = torch.sign(y_tilde) # n
    elif args.y_type == "gaussian":
        y = torch.normal(0, 1, (n,)) # n

    # Generates the test data and labels using the ground-truth regressor.
    X_test = D.sample((args.n_test,)) # n_test x d
    y_tilde_test = X_test @ theta_star # n_test

    # Computes the minimum-norm interpolators for regression and classification.
    Q = torch.linalg.inv(X @ X.T) # n x n

    M = X.T @ Q # n x d
    theta_tilde = M @ y_tilde # d
    theta_hat = M @ y # d

    # Computes the test risk of the minimum-norm interpolators.
    theta_tilde_test_risk = F.mse_loss(
        X_test @ theta_tilde, y_tilde_test)
    theta_hat_test_risk = F.mse_loss(
        X_test @ theta_hat, y_tilde_test)

    theta_tilde_norm = torch.linalg.vector_norm(theta_tilde, ord=2)
    theta_hat_norm = torch.linalg.vector_norm(theta_hat, ord=2)

    # Computes the difference of the test risks of the classifier and regressor.
    risk_diff = (theta_hat_test_risk - theta_tilde_test_risk)

    # Add_range relevant metrics to the results dictionary.
    results["risk_diff"][idx].append(risk_diff)
    results["theta_tilde_risk"][idx].append(theta_tilde_test_risk)
    results["theta_hat_risk"][idx].append(theta_hat_test_risk)
    results["theta_tilde_norm"][idx].append(theta_tilde_norm)
    results["theta_hat_norm"][idx].append(theta_hat_norm)

    # Computes new predictor using theta_hat by taking the top-k indices if
    # theta_star is k-sparse or else by taking the softmax with temperature.
    if args.theta_star_type == "k_sparse":
        selector = topk_inds(theta_hat, args.k_sparse_num)
    elif args.theta_star_type == "step":
        selector = make_selector(args, theta_hat, theta_star)
    elif args.theta_star_type == "unif":
        selector = F.softmax(theta_hat / args.temperature, dim=0)

    y_hat = X @ selector # n
    theta_new = M @ y_hat # d
    theta_new_test_risk = F.mse_loss(
        X_test @ theta_new, y_tilde_test)
    theta_new_norm = torch.linalg.vector_norm(theta_new, ord=2)

    results[f"theta_new_risk"][idx].append(theta_new_test_risk)
    results[f"theta_new_norm"][idx].append(theta_new_norm)

def experiment(args):
    n_range = torch.arange(args.n_start, args.n_end + 1, args.n_step)
    d_range = n_range ** args.d_exponent

    # Initializes the results dictionary.
    results = {
        "risk_diff":        [[] for _ in range(len(n_range))],
        "theta_tilde_risk": [[] for _ in range(len(n_range))],
        "theta_hat_risk":   [[] for _ in range(len(n_range))],
        "theta_new_risk":   [[] for _ in range(len(n_range))],
        "theta_tilde_norm": [[] for _ in range(len(n_range))],
        "theta_hat_norm":   [[] for _ in range(len(n_range))],
        "theta_new_norm":   [[] for _ in range(len(n_range))],
    } 

    # Runs experiment trials.
    for idx, (n, d) in enumerate(zip(n_range, d_range)):
        for t in range(args.trials):
            if t == 0 and n % 100 == 0:
                print(f"Running n={n}...")
            experiment_trial(args, results, idx, n, d)
            
    # Computes the mean over all trials for each metric and value of n in the
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
    parser.add("--d_exponent", default=1.5, type=float)
    parser.add("--k_sparse_num", default=1, type=int)
    parser.add("--n_start", default=20, type=int)
    parser.add("--n_step", default=20, type=int)
    parser.add("--n_end", default=500, type=int)
    parser.add("--n_test", default=100, type=int)
    parser.add("--out_dir", default="out")
    parser.add("--poly_exponent", default=2, type=float)
    parser.add("--spiked_num", default=1, type=int)
    parser.add("--step_val", default=0.5, type=float)
    parser.add("--theta_star_type", choices=["k_sparse", "step", "unif"], default="k_sparse")
    parser.add("--temperature", default=1, type=float)
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

