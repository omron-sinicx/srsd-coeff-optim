import json
from typing import Optional

import sympy


def read_formula(path: str):
    with open(path, "rb") as f:
        formulas = json.load(f)
    return formulas


def read_dataset(path: str):
    with open(path, "rb") as f:
        dataset = [list(map(float, line.split())) for line in f]
    return dataset


def raw_formula_to_skeleton(
    formula: sympy.core.expr.Expr,
) -> tuple[sympy.core.expr.Expr, list, list]:
    """Convert raw formula to skeleton formula

    Args:
        formula (sympy.core.expr.Expr): raw formula

    Returns:
        sympy.core.expr.Expr: skelton formula
        list: list of new variables
        list: list of ground truth coefficient values
    """
    formula = _eval_pi(formula)
    formula = _remove_redundant_numbers(formula)
    return _change_coefficient_to_variable(formula, None, 0)


def _eval_pi(formula: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
    """Replace pi with its numerical value

    Args:
        formula (sympy.core.expr.Expr): sympy formula

    Returns:
        sympy.core.expr.Expr: sympy formula
    """
    return formula.subs(sympy.pi, sympy.pi.evalf())


def _remove_redundant_numbers(formula: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
    """Remove redundant numbers

    Args:
        formula (sympy.core.expr.Expr): formula to be simplified

    Returns:
        sympy.core.expr.Expr: simplified formula
    """
    if formula.is_number:
        # Evaluate if formula does not contain any free variables
        return formula.evalf()
    elif type(formula) is sympy.core.symbol.Symbol:
        return formula
    elif type(formula) in [sympy.core.mul.Mul, sympy.core.add.Add]:
        # Only Add and Mul can have multiple numbers as children
        numbers = []
        non_numbers = []
        for arg in formula.args:
            if arg.is_number:
                numbers.append(arg)
            else:
                non_numbers.append(_remove_redundant_numbers(arg))  # type: ignore
        if len(numbers) == 0:
            return formula.func(*non_numbers)
        else:
            new_number = formula.func(*numbers).evalf()
            return formula.func(new_number, *non_numbers)
    else:
        return formula.func(*[_remove_redundant_numbers(arg) for arg in formula.args])  # type: ignore


def _change_coefficient_to_variable(
    formula: sympy.core.expr.Expr, parent_type: Optional[str], tmp_num: int
) -> tuple[sympy.core.expr.Expr, list, list]:
    """Replace coefficients with variables

    Args:
        formula (sympy.core.expr.Expr): sympy formula
        parent_type (Optional[str]): type of parent node
        tmp_num (int): # of variables already assigned to coefficients

    Returns:
        sympy.core.expr.Expr: sympy formula
        list: list of new variables
        list: list of ground truth coefficient values

    Examples:
        Input: 2*x0*(3+x1)^0.5, None, 0
        Output: c0*x0*(c1+x1)^c2, [c0, c1, c2], [2, 3, 0.5]
    """
    # In x * (-1), -1 is not coefficient
    if parent_type == sympy.core.power.Mul and formula == -1:
        return formula, [], []

    # Numbers under Mul, Add, Pow are "basically" coefficients
    if type(formula) in [
        sympy.core.numbers.Float,
        sympy.core.numbers.Half,
        sympy.core.numbers.Integer,
        sympy.core.numbers.NegativeOne,
        sympy.core.numbers.One,
        sympy.core.numbers.Rational,
    ]:
        if parent_type in [
            sympy.core.mul.Mul,
            sympy.core.add.Add,
            sympy.core.power.Pow,
            None,
        ]:
            c = sympy.symbols("c" + str(tmp_num))
            return c, [c], [float(formula.evalf())]
        else:
            raise ValueError("Call this function after _remove_redundant_numbers")

    # No args
    if len(formula.args) == 0:
        return formula, [], []

    # Recursive call on tree structure
    coefficients = []
    ground_truth = []
    args = formula.args
    new_args = []
    for arg in args:
        (
            sub_formula,
            sub_coefficients,
            sub_ground_truth,
        ) = _change_coefficient_to_variable(
            arg,
            type(formula),
            tmp_num,
        )
        coefficients += sub_coefficients
        ground_truth += sub_ground_truth
        tmp_num += len(sub_ground_truth)
        new_args.append(sub_formula)
    new_formula = type(formula)(*new_args)
    return new_formula, coefficients, ground_truth
