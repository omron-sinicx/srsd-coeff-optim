import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import sympy
from path import RESULT_DIR, SRSD_DIR
from utils import *  # type: ignore

from srsd_coeff_optim import Optimizer  # type: ignore


def optimize(
    data_path: str,
    formula: sympy.Expr,
    optimizer: Optimizer,
    n_data: int,
    init_method: str,
    difficulty: str,
):
    data = read_dataset(data_path)[:n_data]
    formula, _, ground_truth = raw_formula_to_skeleton(formula)

    if init_method == "order":
        rs = np.random.RandomState(0)
        coeff_init = [10 ** rs.uniform(-0.5, 0.5) * x for x in ground_truth]
    elif init_method in ["normal", "uniform"]:
        coeff_init = None
    else:
        raise ValueError("Invalid init_method")
    (
        _,
        status,
        ffinal,
        coeff_opt,
        disc_time_hist,
        cont_time_hist,
        expo_coeffs,
        other_coeffs,
        disc_coeffs,
        cont_coeffs,
        coeff_opt_hist,
        cont_error_hist,
    ) = optimizer.optimize(formula, data, coeff_init, init_method, True)
    res = {
        "difficulty": difficulty,
        "problem": os.path.basename(data_path)[:-4],
        "converted_formula": str(formula),
        "n_coefficients": len(ground_truth),
        "expo_coeffs": expo_coeffs,
        "other_coeffs": other_coeffs,
        "disc_coeffs": disc_coeffs,
        "cont_coeffs": cont_coeffs,
        "ground_truth": ground_truth,
        "real_out_iter": len(disc_time_hist),
        "status": status,
        "final_obj": str(ffinal),
        "coeff_opt": coeff_opt,
        "coeff_opt_hist": coeff_opt_hist,
        "cont_error_hist": cont_error_hist,
        "continuous_time_hist": cont_time_hist,
        "discrete_time_hist": disc_time_hist,
    }
    print(difficulty, os.path.basename(data_path)[:-4])
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["srsd"])
    parser.add_argument("--n_data", type=int)
    parser.add_argument("--discrete_method", choices=["brute-force", "none"])
    parser.add_argument(
        "--continuous_method",
        choices=[
            "bfgs",
            "bfgs-jump",
            "lm",
            "lm-jump",
        ],
    )
    parser.add_argument("--max_contiter", type=int, default=100)
    parser.add_argument("--outiter", type=int)
    parser.add_argument("--beamsize", type=int)
    parser.add_argument("--init_method", choices=["order", "uniform", "normal"])
    parser.add_argument("--n_core", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    optimizer = Optimizer(
        args.discrete_method,
        args.continuous_method,
        args.max_contiter,
        args.outiter,
        args.beamsize,
        args.seed,
        [-1, 0.5, -0.5, 1.5, -1.5, 2, -2, 3, -3, 4, -4, 5, -5],
    )

    # inputs for multiprocessing
    inputs = []
    for difficulty in ["easy", "medium", "hard"]:
        eq_path = f"{SRSD_DIR}/srsd-feynman_{difficulty}/supp_info.json"
        data_dir = f"{SRSD_DIR}/srsd-feynman_{difficulty}/train/"
        formulas = read_formula(eq_path)
        for key in formulas.keys():
            data_path = os.path.join(os.path.dirname(data_dir), key + ".txt")
            formula = sympy.sympify(formulas[key]["sympy_eq_str"])
            inputs.append(
                (
                    data_path,
                    formula,
                    optimizer,
                    args.n_data,
                    args.init_method,
                    difficulty,
                )
            )

    # run multiprocessing
    p = Pool(processes=args.n_core)
    results = p.starmap(optimize, inputs)
    p.close()
    p.join()

    # save results
    all_res = {
        "info": {
            "dataset": args.dataset,
            "n_data": args.n_data,
            "discrete_method": args.discrete_method,
            "continuous_method": args.continuous_method,
            "max_contiter": args.max_contiter,
            "out_iter": args.outiter,
            "beamsize": args.beamsize,
            "init_method": args.init_method,
            "n_core": args.n_core,
            "seed": args.seed,
        }
    }
    for res in results:
        if res["n_coefficients"] == 0:
            continue
        problem = res["problem"]
        del res["problem"]
        all_res[problem] = res
    output_path = "{}/{}/n={}_{}_{}_iter={}_beam={}_{}_seed={}.json".format(
        RESULT_DIR,
        args.dataset,
        args.n_data,
        args.discrete_method,
        args.continuous_method,
        args.outiter,
        args.beamsize,
        args.init_method,
        args.seed,
    )
    with open(output_path, "w") as f:
        json.dump(all_res, f)
