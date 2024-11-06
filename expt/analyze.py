import argparse
import json
import pprint

from path import RESULT_DIR


def read_result(path: str):
    with open(path, "rb") as f:
        data = json.load(f)
    return data


def check_global_optimality(copt, ground_truth):
    """
    Args:
        copt (float): optimized coefficient
        ground_truth (float): ground truth coefficient
    """
    return abs(copt - ground_truth) <= 1e-4 * abs(ground_truth)


def count_result(data, divide_by):
    count = {"total": {}}
    for key in data.keys():
        if key == "info":
            continue
        if type(data[key][divide_by]) is list:
            outkey = len(data[key][divide_by])
        else:
            outkey = data[key][divide_by]
        if outkey not in count:
            count[outkey] = {}
        copt = data[key]["coeff_opt"]
        ground_truth = data[key]["ground_truth"]
        if data[key]["status"] == "failed":
            result = "error"
        else:
            global_optimality = [
                check_global_optimality(c, g) for c, g in zip(copt, ground_truth)
            ]
            if False in global_optimality:
                result = "local optimum"
            else:
                result = "global optimum"
        if result not in count[outkey]:
            count[outkey][result] = 1
        else:
            count[outkey][result] += 1
        if result not in count["total"]:
            count["total"][result] = 1
        else:
            count["total"][result] += 1
    pprint.pprint(count)
    return count


def count_result_by_num_coeff(data):
    count = {"total": {}}
    for key in data.keys():
        if key == "info":
            continue
        outkey = tuple([len(data[key]["expo_coeffs"]), len(data[key]["other_coeffs"])])
        if outkey not in count:
            count[outkey] = {}
        copt = data[key]["coeff_opt"]
        ground_truth = data[key]["ground_truth"]
        if data[key]["status"] == "failed":
            result = "error"
        else:
            global_optimality = [
                check_global_optimality(c, g) for c, g in zip(copt, ground_truth)
            ]
            if False in global_optimality:
                expo_optimal = False
                expo_ground_truth = [
                    x
                    for i, x in enumerate(ground_truth)
                    if "c" + str(i) in data[key]["expo_coeffs"]
                ]
                for coeff in data[key]["coeff_opt_hist"][-1]:
                    expo_coeff = [
                        x
                        for i, x in enumerate(coeff)
                        if "c" + str(i) in data[key]["expo_coeffs"]
                    ]
                    if expo_coeff == expo_ground_truth:
                        expo_optimal = True
                        break
                if expo_optimal:
                    result = "other failed"
                else:
                    result = "expo failed"
            else:
                result = "global optimum"
        if result not in count[outkey]:
            count[outkey][result] = 1
        else:
            count[outkey][result] += 1
        if result not in count["total"]:
            count["total"][result] = 1
        else:
            count["total"][result] += 1
    pprint.pprint(count)
    return count


def check_time(data):
    # only for 1 loop
    global_disc = []
    global_cont = []
    local_disc = []
    local_cont = []
    error_disc = []
    error_cont = []
    for key in data.keys():
        if key == "info":
            continue
        copt = data[key]["coeff_opt"]
        outiter = data[key]["real_out_iter"]
        if data[key]["status"] == "failed":
            error_disc.append(sum(data[key]["discrete_time_hist"]) / outiter)
            error_cont.append(sum(data[key]["continuous_time_hist"]) / outiter)
            continue
        ground_truth = data[key]["ground_truth"]
        global_optimality = [
            check_global_optimality(c, g) for c, g in zip(copt, ground_truth)
        ]
        if False in global_optimality:
            local_disc.append(sum(data[key]["discrete_time_hist"]) / outiter)
            local_cont.append(sum(data[key]["continuous_time_hist"]) / outiter)
        else:
            global_disc.append(sum(data[key]["discrete_time_hist"]) / outiter)
            global_cont.append(sum(data[key]["continuous_time_hist"]) / outiter)
    print(
        "ave_global_disc: ",
        sum(global_disc) / len(global_disc) if len(global_disc) > 0 else "N/A",
    )
    print(
        "ave_global_cont: ",
        sum(global_cont) / len(global_cont) if len(global_cont) > 0 else "N/A",
    )
    print(
        "ave_local_disc: ",
        sum(local_disc) / len(local_disc) if len(local_disc) > 0 else "N/A",
    )
    print(
        "ave_local_cont: ",
        sum(local_cont) / len(local_cont) if len(local_cont) > 0 else "N/A",
    )
    print(
        "ave_error_disc: ",
        sum(error_disc) / len(error_disc) if len(error_disc) > 0 else "N/A",
    )
    print(
        "ave_error_cont: ",
        sum(error_cont) / len(error_cont) if len(error_cont) > 0 else "N/A",
    )
    print(
        "ave_total_disc: ",
        (sum(global_disc) + sum(local_disc) + sum(error_disc))
        / (len(global_disc) + len(local_disc) + len(error_disc)),
    )
    print(
        "ave_total_cont: ",
        (sum(global_cont) + sum(local_cont) + sum(error_cont))
        / (len(global_cont) + len(local_cont) + len(error_cont)),
    )
    print(
        "ave_total: ",
        (sum(global_disc) + sum(local_disc) + sum(error_disc))
        / (len(global_disc) + len(local_disc) + len(error_disc))
        + (sum(global_cont) + sum(local_cont) + sum(error_cont))
        / (len(global_cont) + len(local_cont) + len(error_cont)),
    )


def count_error(data):
    messages = {}
    for key in data.keys():
        if key == "info":
            continue
        if data[key]["status"] == "failed":
            l = data[key]["cont_error_hist"]
            for i in range(len(l)):
                for j in range(len(l[i])):
                    if l[i][j] not in messages:
                        messages[l[i][j]] = 1
                    else:
                        messages[l[i][j]] += 1
    pprint.pprint(messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_method",
        choices=[
            "stats_by_num_coeff",
            "stats_by_difficulty",
            "time",
            "count_error",
        ],
    )
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
    parser.add_argument("--outiter", type=int)
    parser.add_argument("--beamsize", type=int)
    parser.add_argument("--init_method", choices=["order", "uniform", "normal"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    file = "n={}_{}_{}_iter={}_beam={}_{}_seed={}.json".format(
        args.n_data,
        args.discrete_method,
        args.continuous_method,
        args.outiter,
        args.beamsize,
        args.init_method,
        args.seed,
    )
    path = f"{RESULT_DIR}/{args.dataset}/{file}"
    print(file)
    data = read_result(path)
    if args.eval_method == "stats_by_num_coeff":
        count_result_by_num_coeff(data)
    elif args.eval_method == "stats_by_difficulty":
        count_result(data, "difficulty")
    elif args.eval_method == "time":
        check_time(data)
    elif args.eval_method == "count_error":
        count_error(data)
