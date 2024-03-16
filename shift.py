from configargparse import Parser
from distutils.util import strtobool
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

plt.rcParams["text.usetex"] = True

def repeated_argmax(x, k):
    y = torch.zeros(len(x))
    argmaxes = torch.topk(x, k)[1]
    y[argmaxes] = 1
    return y

def set_display_metrics(args, fname):
    # Adds y_type to filename.
    if args.y_type == "sgn":
        fname += "_sgn"
        display_y_type = r"$y=sgn(\tilde{y})$"
    elif args.y_type == "gaussian":
        fname += "_gaussian"
        display_y_type = r"$y\sim\mathcal{N}(0,1)$"

    # Adds theta_star_type to filename.
    if args.theta_star_type == "unif":
        fname += "_unif"
        display_theta_star_type = r"$\theta^\star\sim U(-1,1)$"
    elif args.theta_star_type == "ei":
        vecs = "".join([str(j) for j in range(1, args.theta_star_ei_num + 1)])
        fname += f"_e{vecs}"
        display_theta_star_type = fr"$\theta^\star=e_{{{vecs}}}$"

    # Adds cov_type to filename.
    if args.cov_type == "isotropic":
        fname += "_isotropic.png"
    elif args.cov_type == "spiked":
        fname += "_spiked.png"
    elif args.cov_type == "poly":
        fname += "_poly.png"

    fname = os.path.join(args.out_dir, fname)

    return fname, display_theta_star_type, display_y_type

def plot(args, results):
    d_range = np.arange(args.d_start, args.d_end + 1, args.d_step)

    # Defines a helper function for plotting the results.
    def plot_helper(
        metrics, display_metrics, fname, xmin=None, xmax=None, ymin=0, ymax=None
    ):
        fname, display_theta_star_type, display_y_type = \
                set_display_metrics(args, fname)

        start = (xmin - args.d_start) // args.d_step if xmin else 0
        end = (xmax - args.d_start) // args.d_step if xmax else len(d_range)

        # Plots specified range of data.
        for name, display_name in zip(metrics, display_metrics):
            if args.cov_type == "spiked":
                label = display_name
                label += rf" ratio=${args.spiked_ratio}$,"
                label += rf" num=${args.spiked_num}$"
                plt.plot(
                    d_range[start:end],
                    results[name][start:end],
                    label=label,
                )
            else:
                if args.cov_type == "isotropic":
                    label = display_name
                elif args.cov_type == "poly":
                    label = display_name
                    label += rf" poly=${args.poly_exponent}$"

                plt.plot(
                    d_range[start:end],
                    results[name][start:end],
                    label=label,
                )

        # Automatically crops the y-axis if a bound on the x-axis is specified.
        if (xmin or xmax) and not ymax:
            ymax = max(torch.tensor([
                result[metric][start:end] for metric in metrics
            ]).flatten())

        title = fr"{display_theta_star_type}, {display_y_type}, "
        title += fr"{args.cov_type}, $n={args.n}$, "
        title += fr"$\|\theta^\star\|_2^2={round(args.var, 3)}$, "
        title += f"{args.trials} trials"

        plt.ylim(ymin, ymax)
        plt.legend()
        plt.title(title)
        plt.xlabel("d")
        plt.savefig(fname, bbox_inches="tight", dpi=600)
        plt.clf()

    if args.y_type == "sgn":
        ymax = args.var
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

def experiment_trial(args, results, idx, d):
    # Generates the ground-truth regressor.
    if args.theta_star_type == "unif":
        theta_star = -2 * torch.random(d) + 1 # theta_star ~ Unif(-1, 1).
        theta_star = theta_star / torch.norm(theta_star)
    elif args.theta_star_type == "ei":
        theta_star = torch.zeros(d)
        theta_star[:args.theta_star_ei_num] = 1
    args.var = torch.norm(theta_star).item()

    # Generates the covariance matrix.
    if args.cov_type == "isotropic":
        # Generates an isotropic (identity) covariance matrix.
        cov = torch.eye(d)
    elif args.cov_type == "spiked":
        # Generates a spiked covariance matrix which is diagonal with 1 in the
        # first spiked_num entries and 1 / spiked_ratio in the rest.
        cov = torch.eye(d)
        for j in range(min(args.spiked_num, d)):
            if args.spiked_ratio == "d^2":
                cov[j, j] = d ** 2
            elif args.spiked_ratio == "d":
                cov[j, j] = d
            else:
                cov[j, j] = args.spiked_ratio
        cov = cov / cov[0, 0]
    elif args.cov_type == "poly":
        # Generates a polynomial decay covariance matrix which is diagonal
        # with the jth entry being j^{-poly_exponent}.
        eigenvalues = torch.tensor([
            j ** -args.poly_exponent for j in range(1, d + 1)
        ])
        cov = torch.diag(eigenvalues)

    # Generates the train data and labels using the ground-truth regressor.
    D = MultivariateNormal(torch.zeros(d), cov)
    X = D.sample((args.n,)).T
    y_tilde = X.T @ theta_star

    # Generates classifier data as either the signs of the regression labels
    # or as independent standard Gaussians.
    if args.y_type == "sgn":
        y = torch.sign(y_tilde)
    elif args.y_type == "gaussian":
        y = torch.normal(0, 1, (args.n,))

    # Generates the test data and labels using the ground-truth regressor.
    X_test = D.sample((args.n_test,)).T
    y_tilde_test = X_test.T @ theta_star

    # Computes the minimum-norm interpolators for regression and classification.
    M = X.T @ torch.linalg.pinv(X @ X.T, hermitian=True)
    theta_tilde = M.T @ y_tilde
    theta_hat = M.T @ y

    # Computes the test risk of the minimum-norm interpolators.
    theta_tilde_test_risk = F.mse_loss(
        X_test.T @ theta_tilde, y_tilde_test)
    theta_hat_test_risk = F.mse_loss(
        X_test.T @ theta_hat, y_tilde_test)

    theta_tilde_norm = torch.norm(theta_tilde)
    theta_hat_norm = torch.norm(theta_hat)

    # Computes the difference of the test risks of the classifier and regressor.
    risk_diff = (theta_hat_test_risk - theta_tilde_test_risk)

    # Add_range relevant metrics to the results dictionary.
    results["risk_diff"][idx].append(risk_diff)
    results["theta_tilde_risk"][idx].append(theta_tilde_test_risk)
    results["theta_hat_risk"][idx].append(theta_hat_test_risk)
    results["theta_tilde_norm"][idx].append(theta_tilde_norm)
    results["theta_hat_norm"][idx].append(theta_hat_norm)

    # Computes new predictor with repeated argmax strategy.
    selector = repeated_argmax(theta_hat, args.theta_star_ei_num)
    y_hat = X.T @ selector
    theta_new = M.T @ y_hat
    theta_new_test_risk = F.mse_loss(
        X_test.T @ theta_new, y_tilde_test)
    theta_new_norm = torch.norm(theta_new)

    results[f"theta_new_risk"][idx].append(theta_new_test_risk)
    results[f"theta_new_norm"][idx].append(theta_new_norm)

def experiment(args):
    d_range = torch.arange(args.d_start, args.d_end + 1, args.d_step)

    # Initializes the results dictionary.
    results = {
        "risk_diff":        [[] for _ in range(len(d_range))],
        "theta_tilde_risk": [[] for _ in range(len(d_range))],
        "theta_hat_risk":   [[] for _ in range(len(d_range))],
        "theta_new_risk":   [[] for _ in range(len(d_range))],
        "theta_tilde_norm": [[] for _ in range(len(d_range))],
        "theta_hat_norm":   [[] for _ in range(len(d_range))],
        "theta_new_norm":   [[] for _ in range(len(d_range))],
    } 

    # Runs experiment trials.
    for idx, d in enumerate(d_range):
        for _ in range(args.trials):
            experiment_trial(args, results, idx, d)
            
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

    # Load_range configuration parameters into parser.
    parser.add("--cov_type", choices=["isotropic", "poly", "spiked"], default="spiked")
    parser.add("--cuda", default=True, type=lambda x: bool(strtobool(x)))
    parser.add("--d_start", default=5, type=int)
    parser.add("--d_step", default=5, type=int)
    parser.add("--d_end", default=1000, type=int)
    parser.add("--poly_exponent", default=2, type=float)
    parser.add("--n", default=100, type=int)
    parser.add("--n_test", default=100, type=int)
    parser.add("--out_dir", default="out")
    parser.add("--spiked_num", default=1, type=int)
    parser.add("--spiked_ratio", default="d")
    parser.add("--theta_star_type", choices=["ei", "unif"], default="ei")
    parser.add("--theta_star_ei_num", default=1, type=int)
    parser.add("--trials", default=10, type=int)
    parser.add("--y_type", choices=["gaussian", "sgn"], default="sgn")
    args = parser.parse_args()

    # Spiked ratio is allowed to be integer (e.g., 3) or string (e.g., "d").
    allowed_spiked_ratio_strings = ["d", "d^2"]
    try:
        args.spiked_ratio = int(args.spiked_ratio)
    except ValueError:
        if args.spiked_ratio not in allowed_spiked_ratio_strings:
            raise ValueError((
                "If spiked_ratio is a string, it must be one of"
                f" {allowed_spiked_ratio_strings}."
            ))

    main(args)

