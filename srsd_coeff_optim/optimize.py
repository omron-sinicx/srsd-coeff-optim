"""classes for optimization

inheritance structure:

    DiscreteOptimizer
    |---BruteForceOptimizer

    ContinuousOptimizer
    |---BFGSOptimizer
    |---BFGSJumpOptimizer
    |---LMOptimizer
    |---LMJumpOptimizer

    Optimizer
"""

import heapq
import math
import time
import warnings
from typing import Callable, Optional

import numpy as np
import scipy  # type: ignore
import scipy.optimize  # type: ignore
import sympy

from .formula import Formula  # type: ignore


class DiscreteOptimizer:
    """class for discrete optimizer

    Attributes:
        max_num_returns (int): maximum # of candidates to return
        candidates (list[float]): candidate values for discrete coefficients
        num_cand (int): # of candidate values for discrete coefficients
    """

    def __init__(self, max_num_returns: int, candidates: list[float]) -> None:
        """initialization

        Args:
            max_num_returns (int): maximum # of candidates to return
            candidates (list[float]): candidate values for discrete coefficients

        Raises:
            AssertionError: if max_num_returns is not positive
        """
        assert max_num_returns > 0, "max_num_returns must be positive"
        self.max_num_returns: int = max_num_returns
        self.candidates: list[float] = candidates
        self.num_cand: int = len(candidates)

    def optimize(
        self,
        exprlist: list[sympy.core.expr.Expr],
        symbols: list[sympy.core.symbol.Symbol],
    ) -> list[tuple[list[float], int]]:
        """return the best coefficients by discrete search

        Args:
            exprlist (list[sympy.core.expr.Expr]): list of objective functions
            symbols (list[sympy.core.symbol.Symbol]): list of discrete coefficients

        Returns:
            list[tuple[list[float], int]]: list of (coefficients, index of function in funclist).

        Note:
            Returned list l satisfies len(l) <= min(max_num_returns, num_cand ** len(symbols))
        """
        raise NotImplementedError


class BruteForceOptimizer(DiscreteOptimizer):
    """class for brute-force optimizer"""

    def __init__(self, max_num_returns: int, candidates: list[float]) -> None:
        """initialization"""
        super().__init__(max_num_returns, candidates)

    def optimize(
        self,
        exprlist: list[sympy.core.expr.Expr],
        symbols: list[sympy.core.symbol.Symbol],
    ) -> list[tuple[list[float], int]]:
        """find the best coefficients by brute-force search"""
        # convert sympy expression to function
        funclist: list[Callable] = [
            sympy.lambdify(symbols, e, "math") for e in exprlist
        ]

        # brute-force search
        n_args = len(symbols)
        min_fval = [(float("inf"), -1) for _ in range(self.num_cand**n_args)]
        for idx, f in enumerate(funclist):
            for candidx in range(self.num_cand**n_args):
                args: list[float] = []
                tmp = candidx
                for _ in range(n_args):
                    args.append(self.candidates[tmp % self.num_cand])
                    tmp //= self.num_cand
                try:
                    fval = f(*args)
                except:
                    # ignore errors
                    fval = float("inf")
                if type(fval) is complex:
                    fval = float("inf")
                if fval < min_fval[candidx][0]:
                    min_fval[candidx] = (fval, idx)

        # select max_num_returns best candidx
        fval_xopt_idx: list[tuple[float, int, int]] = []
        heapq.heapify(fval_xopt_idx)
        for candidx in range(self.num_cand**n_args):
            if len(fval_xopt_idx) < self.max_num_returns:
                heapq.heappush(
                    fval_xopt_idx,
                    (-min_fval[candidx][0], candidx, min_fval[candidx][1]),
                )
            elif min_fval[candidx][0] < -fval_xopt_idx[0][0]:
                heapq.heappop(fval_xopt_idx)
                heapq.heappush(
                    fval_xopt_idx,
                    (-min_fval[candidx][0], candidx, min_fval[candidx][1]),
                )

        # convart candidx to tuple of variable values
        ans: list[tuple[list[float], int]] = []
        while True:
            try:
                _, candidx, idx = heapq.heappop(fval_xopt_idx)
                args = []
                for _ in range(n_args):
                    args.append(self.candidates[candidx % self.num_cand])
                    candidx //= self.num_cand
                ans.append((args, idx))
            except:
                break
        return ans


class ContinuousOptimizer:
    """class for continuous optimizer

    Attributes:
        maxiter (int): maximum # of iterations
    """

    def __init__(self, maxiter: int) -> None:
        """initialization

        Args:
            maxiter (int): maximum # of iterations

        Raises:
            AssertionError: if maxiter is not positive
        """
        assert maxiter > 0, "maxiter must be positive"
        self.maxiter = maxiter


class BFGSOptimizer(ContinuousOptimizer):
    """class for bfgs optimizer"""

    def __init__(self, maxiter: int) -> None:
        """initialization"""
        super().__init__(maxiter)

    def optimize(
        self,
        expr: sympy.core.expr.Expr,
        symbols: list[sympy.core.symbol.Symbol],
        x0: list[float],
        expr_prime: list[sympy.core.expr.Expr],
    ) -> tuple[list[float], Optional[float], str]:
        """find the best coefficients by BFGS optimization

        Args:
            expr (sympy.core.expr.Expr): objective function
            symbols (list[sympy.core.symbol.Symbol]): list of continuous coefficients
            x0 (list[float]): initial value for continuous coefficients
            expr_prime (list[sympy.core.expr.Expr]): gradient of objective function

        Returns:
            tuple[list[float], Optional[float], str]:
            (final value of coefficients, final value of objective function, status)
        """
        # convert sympy expression to function
        f_multiargs: Callable = sympy.lambdify(symbols, expr, "math")
        f: Callable[[*list[float]], float] = lambda x: f_multiargs(*x)  # type: ignore
        fprime_multiargs: Callable = sympy.lambdify(symbols, expr_prime, "math")
        fprime: Callable[[*list[float]], list[float]] = lambda x: fprime_multiargs(*x)  # type: ignore

        # optimize
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", scipy.optimize.OptimizeWarning)
        try:
            xopt_np, fopt, _, _, _, _, _, _ = scipy.optimize.fmin_bfgs(
                f,
                x0,
                fprime,
                maxiter=self.maxiter,
                full_output=True,
                disp=False,
                retall=True,
            )
            xopt = xopt_np.tolist()
            warnings.resetwarnings()
            return xopt, fopt, "converged"
        except Exception as e:
            warnings.resetwarnings()
            return x0, None, str(e)


class LMOptimizer(ContinuousOptimizer):
    """class for Levenberg Marquardt optimizer"""

    def __init__(self, maxiter: int) -> None:
        """initialization"""
        super().__init__(maxiter)

    def optimize(
        self,
        data: list[list[float]],
        formula: sympy.core.expr.Expr,
        symbols: list[sympy.core.symbol.Symbol],
        variables: list[sympy.core.symbol.Symbol],
        x0: list[float],
    ) -> tuple[list[float], Optional[float], str]:
        """find the best coefficients by Levenberg Marquardt optimization

        Args:
            data (list[list[float]]): dataset
            formula (sympy.core.expr.Expr): skeleton with discrete coefficients fixed
            symbols (list[sympy.core.symbol.Symbol]): list of continuous coefficients
            variables (list[sympy.core.symbol.Symbol]): list of variables [x0, x1, ...]
            x0 (list[float]): initial value for continuous coefficients

        Returns:
            tuple[list[float], Optional[float], str]:
            (final value of coefficients, final value of objective function, status)
        """
        x = np.array(data)[:, :-1]
        y = np.array(data)[:, -1]

        def residual(c: np.ndarray, xarg: np.ndarray, yarg: np.ndarray) -> np.ndarray:
            partial = sympy.lambdify(
                variables,
                formula.subs([(s, c[i]) for i, s in enumerate(symbols)]),
                "numpy",
            )
            return yarg - np.apply_along_axis(
                lambda row: partial(*row),
                1,
                xarg,
            )

        def jacobian(c: np.ndarray, xarg: np.ndarray, yarg: np.ndarray) -> np.ndarray:
            partial_prime = [
                sympy.lambdify(
                    variables,
                    sympy.diff(formula, sp).subs(
                        [(s, c[i]) for i, s in enumerate(symbols)]
                    ),
                    "numpy",
                )
                for sp in symbols
            ]
            return np.apply_along_axis(
                lambda row: [-partial_prime[i](*row) for i in range(len(c))],
                1,
                xarg,
            )

        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", scipy.optimize.OptimizeWarning)
        try:
            xopt_np, _, infodict, mes, _ = scipy.optimize.leastsq(
                residual,
                np.array(x0),
                args=(x, y),
                Dfun=jacobian,
                full_output=True,
                maxfev=self.maxiter,
            )
            xopt = xopt_np.tolist()
            fopt = np.sum(infodict["fvec"] ** 2) / len(x)
            warnings.resetwarnings()
            return xopt, fopt, mes
        except Exception as e:
            warnings.resetwarnings()
            return x0, None, str(e)


def _solve_y_fc(y: float | sympy.core.numbers.Float, f: sympy.core.expr.Expr) -> float:
    """solve y = f(c) for c

    Args:
        y (float):
        f (sympy.core.expr.Expr):

    Returns:
        float: c

    Raises:
        ValueError: if invalid function
    """
    if f.is_symbol:
        return y  # type: ignore
    elif type(f) is sympy.core.add.Add:
        if f.args[0].is_number:
            return _solve_y_fc(y - f.args[0].evalf(), f.args[1])
        else:
            return _solve_y_fc(y - f.args[1].evalf(), f.args[0])
    elif type(f) is sympy.core.mul.Mul:
        if f.args[0].is_number:
            return _solve_y_fc(y / f.args[0].evalf(), f.args[1])
        else:
            return _solve_y_fc(y / f.args[1].evalf(), f.args[0])
    elif type(f) is sympy.core.power.Pow:
        if f.args[0].is_number:
            return _solve_y_fc(math.log(y, f.args[0].evalf()), f.args[1])
        else:
            return _solve_y_fc(y ** (1 / f.args[1].evalf()), f.args[0])
    elif type(f) is sympy.functions.elementary.exponential.log:
        return _solve_y_fc(math.exp(y), f.args[0])
    elif type(f) is sympy.functions.elementary.exponential.exp:
        return _solve_y_fc(math.log(y), f.args[0])
    elif type(f) is sympy.functions.elementary.hyperbolic.tanh:
        return _solve_y_fc(math.atanh(y), f.args[0])
    elif type(f) is sympy.functions.elementary.trigonometric.sin:  # compromise
        return _solve_y_fc(math.asin(y), f.args[0])
    elif type(f) is sympy.functions.elementary.trigonometric.cos:  # compromise
        return _solve_y_fc(math.acos(y), f.args[0])
    else:
        raise ValueError("Invalid function")


class BFGSJumpOptimizer(ContinuousOptimizer):
    """class for bfgs optimizer with jump restart"""

    def __init__(self, maxiter: int, maxrestart: int) -> None:
        """initialization"""
        super().__init__(maxiter)
        self.maxrestart = maxrestart

    def optimize(
        self,
        expr: sympy.core.expr.Expr,
        formula: sympy.core.expr.Expr,
        symbols: list[sympy.core.symbol.Symbol],
        variables: list[sympy.core.symbol.Symbol],
        x0: list[float],
        expr_prime: list[sympy.core.expr.Expr],
        data: list[list[float]],
    ) -> tuple[list[float], Optional[float], str]:
        """find the best coefficients by BFGS optimization with jump restart

        Args:
            expr (sympy.core.expr.Expr): objective function
            formula (sympy.core.expr.Expr): formula
            symbols (list[sympy.core.symbol.Symbol]): list of continuous coefficients
            variables (list[sympy.core.symbol.Symbol]): list of variables [x0, x1, ...]
            x0 (list[float]): initial value for continuous coefficients
            expr_prime (list[sympy.core.expr.Expr]): gradient of objective function
            data (list[list[float]]): dataset

        Returns:
            tuple[list[float], Optional[float], str]:
            (final value of coefficients, final value of objective function, status)
        """
        # convert sympy expression to function
        f_multiargs: Callable = sympy.lambdify(symbols, expr, "math")
        f: Callable[[*list[float]], float] = lambda x: f_multiargs(*x)  # type: ignore
        fprime_multiargs: Callable = sympy.lambdify(symbols, expr_prime, "math")
        fprime: Callable[[*list[float]], list[float]] = lambda x: fprime_multiargs(*x)  # type: ignore

        # optimize
        n_coeffs = len(x0)
        coeff_tmp = x0.copy()
        for _ in range(self.maxrestart):
            ###################
            # BFGS optimization
            ###################
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", scipy.optimize.OptimizeWarning)
            try:
                xopt_np, fopt, _, _, _, _, _, _ = scipy.optimize.fmin_bfgs(
                    f,
                    coeff_tmp,
                    fprime,
                    maxiter=self.maxiter,
                    full_output=True,
                    disp=False,
                    retall=True,
                )
                xopt = xopt_np.tolist()
                warnings.resetwarnings()
            except Exception as e:
                warnings.resetwarnings()
                return coeff_tmp, None, str(e)
            ######
            # jump
            ######
            max_dec: float = -1.0
            max_idx = -1
            max_val: float = float("inf")
            for i in range(n_coeffs):
                # fix symbols[j] (j != i)
                fixed_formula = formula.subs(
                    [(symbols[j], xopt[j]) for j in range(n_coeffs) if j != i]
                )
                # optimize symbols[i]
                s: float = 0
                cnt: int = 0
                for row in data:
                    try:
                        s += float(
                            _solve_y_fc(
                                row[-1],
                                fixed_formula.subs(
                                    [
                                        (variables[j], row[j])
                                        for j in range(len(row) - 1)
                                    ]
                                ),
                            )
                        )
                        cnt += 1
                    except:
                        pass
                # how much fopt decreases if symbols[i] is replaced by s / cnt
                if cnt == 0:
                    continue
                try:
                    dec = fopt - f(xopt[:i] + [s / cnt] + xopt[i + 1 :])
                except:
                    continue
                if type(dec) is complex:
                    continue
                if dec > 0 and dec > max_dec:
                    if s / cnt == 0 and xopt[i] == 0:
                        continue
                    if abs(s / cnt - xopt[i]) / max(abs(s / cnt), abs(xopt[i])) < 1e-4:
                        continue
                    max_dec = dec
                    max_idx = i
                    max_val = s / cnt
            if max_dec != -1:
                coeff_tmp = xopt.copy()
                coeff_tmp[max_idx] = max_val
            else:
                return xopt, fopt, "global-converged"
        return coeff_tmp, fopt - max_dec, "converged"


class LMJumpOptimizer(ContinuousOptimizer):
    """class for Levenberg Marquardt optimizer with jump restart"""

    def __init__(self, maxiter: int, maxrestart: int) -> None:
        """initialization"""
        super().__init__(maxiter)
        self.maxrestart = maxrestart

    def optimize(
        self,
        expr: sympy.core.expr.Expr,
        formula: sympy.core.expr.Expr,
        symbols: list[sympy.core.symbol.Symbol],
        variables: list[sympy.core.symbol.Symbol],
        x0: list[float],
        data: list[list[float]],
    ) -> tuple[list[float], Optional[float], str]:
        """find the best coefficients by LM optimization with jump restart

        Args:
            expr (sympy.core.expr.Expr): objective function
            formula (sympy.core.expr.Expr): formula
            symbols (list[sympy.core.symbol.Symbol]): list of continuous coefficients
            variables (list[sympy.core.symbol.Symbol]): list of variables [x0, x1, ...]
            x0 (list[float]): initial value for continuous coefficients
            data (list[list[float]]): dataset

        Returns:
            tuple[list[float], Optional[float], str]:
            (final value of coefficients, final value of objective function, status)
        """
        # convert sympy expression to function
        f_multiargs: Callable = sympy.lambdify(symbols, expr, "math")
        f: Callable[[*list[float]], float] = lambda x: f_multiargs(*x)  # type: ignore

        # preparation for LM optimization
        x = np.array(data)[:, :-1]
        y = np.array(data)[:, -1]

        def residual(c: np.ndarray, xarg: np.ndarray, yarg: np.ndarray) -> np.ndarray:
            partial = sympy.lambdify(
                variables,
                formula.subs([(s, c[i]) for i, s in enumerate(symbols)]),
                "numpy",
            )
            return yarg - np.apply_along_axis(
                lambda row: partial(*row),
                1,
                xarg,
            )

        def jacobian(c: np.ndarray, xarg: np.ndarray, yarg: np.ndarray) -> np.ndarray:
            partial_prime = [
                sympy.lambdify(
                    variables,
                    sympy.diff(formula, sp).subs(
                        [(s, c[i]) for i, s in enumerate(symbols)]
                    ),
                    "numpy",
                )
                for sp in symbols
            ]
            return np.apply_along_axis(
                lambda row: [-partial_prime[i](*row) for i in range(len(c))],
                1,
                xarg,
            )

        # optimize
        n_coeffs = len(x0)
        coeff_tmp = x0.copy()
        for _ in range(self.maxrestart):
            ###################
            # LM optimization
            ###################
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", scipy.optimize.OptimizeWarning)
            try:
                xopt_np, _, infodict, mes, _ = scipy.optimize.leastsq(
                    residual,
                    np.array(x0),
                    args=(x, y),
                    Dfun=jacobian,
                    full_output=True,
                    maxfev=self.maxiter,
                )
                xopt = xopt_np.tolist()
                fopt = np.sum(infodict["fvec"] ** 2) / len(x)
                warnings.resetwarnings()
            except Exception as e:
                warnings.resetwarnings()
                return x0, None, str(e)
            ######
            # jump
            ######
            max_dec: float = -1.0
            max_idx = -1
            max_val: float = float("inf")
            for i in range(n_coeffs):
                # fix symbols[j] (j != i)
                fixed_formula = formula.subs(
                    [(symbols[j], xopt[j]) for j in range(n_coeffs) if j != i]
                )
                # optimize symbols[i]
                s: float = 0
                cnt: int = 0
                for row in data:
                    try:
                        s += float(
                            _solve_y_fc(
                                row[-1],
                                fixed_formula.subs(
                                    [
                                        (variables[j], row[j])
                                        for j in range(len(row) - 1)
                                    ]
                                ),
                            )
                        )
                        cnt += 1
                    except:
                        pass
                # how much fopt decreases if symbols[i] is replaced by s / cnt
                if cnt == 0:
                    continue
                try:
                    dec = fopt - f(xopt[:i] + [s / cnt] + xopt[i + 1 :])
                except:
                    continue
                if type(dec) is complex:
                    continue
                if dec > 0 and dec > max_dec:
                    if s / cnt == 0 and xopt[i] == 0:
                        continue
                    if abs(s / cnt - xopt[i]) / max(abs(s / cnt), abs(xopt[i])) < 1e-4:
                        continue
                    max_dec = dec
                    max_idx = i
                    max_val = s / cnt
            if max_dec != -1:
                coeff_tmp = xopt.copy()
                coeff_tmp[max_idx] = max_val
            else:
                return xopt, fopt, "global-converged"
        return coeff_tmp, fopt - max_dec, "converged"


class Optimizer:
    """class for optimizer

    Attributes:
        discrete_method (str): method for discrete optimization
        continuous_method (str): method for continuous optimization
        max_contiter (int): maximum # of iterations for gradient-based optimization
        outiter (int): maximum # of iterations for outer loop
        beamsize (int): beam size
        seed (int): seed value
        discrete_optimizer (Optional[DiscreteOptimizer]): discrete optimizer
        continuous_optimizer (ContinuousOptimizer): continuous optimizer
    """

    def __init__(
        self,
        discrete_method: str,
        continuous_method: str,
        max_contiter: int,
        outiter: int,
        beamsize: int,
        seed: int,
        discrete_candidates: Optional[list[float]] = None,
    ) -> None:
        """initialization

        Args:
            discrete_method (str): method for discrete optimization
            continuous_method (str): method for continuous optimization
            max_contiter (int): maximum # of iterations for gradient-based optimization
            outiter (int): maximum # of iterations for outer loop
            beamsize (int): beam size
            seed (int): seed value
            discrete_candidates (Optional[list[float]]): candidate values for discrete coefficients

        Raises:
            AssertionError: if invalid arguments are given
        """
        assert discrete_method in [
            "brute-force",
            "none",
        ], "Invalid discrete method"
        assert continuous_method in [
            "bfgs",
            "bfgs-jump",
            "lm",
            "lm-jump",
        ], "Invalid continuous method"
        assert max_contiter > 0, "max_contiter must be positive"
        assert outiter > 0, "outiter must be positive"
        assert beamsize > 0, "beamsize must be positive"
        assert (
            discrete_method != "none" or outiter == 1
        ), "outiter must be 1 if discrete_method is none"
        self.discrete_method = discrete_method
        self.continuous_method = continuous_method
        self.max_contiter = max_contiter
        self.max_contrestart = 10
        self.outiter = outiter
        self.beamsize = beamsize
        self.seed = seed

        if self.discrete_method == "brute-force":
            assert (
                discrete_candidates is not None
            ), "discrete_candidates must be given for brute-force method"
            self.discrete_optimizer: Optional[DiscreteOptimizer] = BruteForceOptimizer(
                self.beamsize, discrete_candidates
            )
        else:
            self.discrete_optimizer = None
        if self.continuous_method == "bfgs":
            self.continuous_optimizer: ContinuousOptimizer = BFGSOptimizer(
                self.max_contiter
            )
        elif self.continuous_method == "bfgs-jump":
            self.continuous_optimizer = BFGSJumpOptimizer(
                self.max_contiter, self.max_contrestart
            )
        elif self.continuous_method == "lm":
            self.continuous_optimizer = LMOptimizer(self.max_contiter)
        elif self.continuous_method == "lm-jump":
            self.continuous_optimizer = LMJumpOptimizer(
                self.max_contiter, self.max_contrestart
            )

    def _merge(
        self, disc: list[float], cont: list[float], formula: Formula
    ) -> list[float]:
        """merge discrete and continuous coefficients

        Args:
            disc (list[float]): discrete coefficients
            cont (list[float]): continuous coefficients
            formula (Formula): formula

        Returns:
            list[float]: merged coefficients
        """
        merged = [float("inf") for _ in range(len(disc) + len(cont))]
        idx = 0
        for i in range(len(merged)):
            if "c" + str(i) in formula.disc_coeffs_name:
                merged[i] = disc[idx]
                idx += 1
        idx = 0
        for i in range(len(merged)):
            if "c" + str(i) in formula.cont_coeffs_name:
                merged[i] = cont[idx]
                idx += 1
        return merged

    def optimize(
        self,
        skelton: str | sympy.core.expr.Expr,
        data: list[list[float]],
        coeff_init: Optional[list[float]] = None,
        init_method: str = "normal",
        allinfo=False,
    ) -> (
        tuple[sympy.core.expr.Expr, str]
        | tuple[
            sympy.core.expr.Expr,
            str,
            float,
            list[float],
            list[float],
            list[float],
            list[str],
            list[str],
            list[str],
            list[str],
            list[list[list[float]]],
            list[list[str]],
        ]
    ):
        """optimize the coefficients in skelton using data

        Args:
            skelton (str | sympy.core.expr.Expr): formula
            data (list[list[float]]): dataset
            coeff_init (Optional[list[float]]): initial value for coefficients
            init_method (str): method for coefficient initialization
            allinfo (bool): if True, return all information

        Returns:
            tuple[sympy.core.expr.Expr, str] |
            tuple[
                sympy.core.expr.Expr,
                str,
                float,
                list[float],
                list[float],
                list[float],
                list[str],
                list[str],
                list[str],
                list[str],
                list[list[list[float]]],
                list[list[str]]]:
            (final value of formula, status) |
            (final value of formula, status, final value of objective function, final value of coefficients,
            time history for discrete optimization, time history for continuous optimization,
            list of exponential coefficients, list of other coefficients,
            list of discrete coefficients, list of continuous coefficients,
            history of optimized coefficient values, history of errors)
        """
        # fix seed
        np.random.seed(self.seed)
        rs = np.random.RandomState(self.seed)

        # Convert skelton to sympy expression
        formula = Formula(skelton)

        # No coefficients
        if len(formula.expo_coeffs) + len(formula.other_coeffs) == 0:
            if allinfo:
                formula.formula, "succeeded",
            else:
                return (
                    formula.formula,
                    "succeeded",
                    0,
                    [],
                    [0],
                    [0],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )

        # Objective function
        formula.set_objective_function(data)

        # Separate coefficients into discrete and continuous
        formula.set_disc_cont_coeffs(self.discrete_optimizer is not None)

        # Variables for optimization
        if coeff_init is None:
            assert init_method in ["normal", "uniform"], "Invalid init_method"
            init_num = self.beamsize if self.discrete_method == "none" else 1
            if init_method == "normal":
                cont_opts: list[list[float]] = [
                    [rs.normal(0, 1) for _ in range(len(formula.cont_coeffs))]
                    for _ in range(init_num)
                ]
            elif init_method == "uniform":
                cont_opts = [
                    [rs.uniform(0, 1) for _ in range(len(formula.cont_coeffs))]
                    for _ in range(init_num)
                ]
        else:
            cont_opts = [
                [
                    coeff_init[i]
                    for i in range(len(formula.disc_coeffs + formula.cont_coeffs))
                    if "c" + str(i) in formula.cont_coeffs_name
                ]
            ]
        disc_opts_cont_init: list[tuple[list[float], list[float]]] = []
        fval_disc_cont_opts: list[tuple[float, list[float], list[float]]] = []

        # Optimize
        disc_time_hist: list[float] = []
        cont_time_hist: list[float] = []
        coeff_opt_hist: list[list[list[float]]] = []
        cont_error_hist: list[list[str]] = []
        for _ in range(self.outiter):
            #######################
            # Discrete optimization
            #######################
            disc_opts_cont_init.clear()
            if self.discrete_optimizer is None or len(formula.disc_coeffs) == 0:
                # Only continuous coefficients
                for cont_opt in cont_opts:
                    disc_opts_cont_init.append(([], cont_opt))
                disc_time_hist.append(0)
            else:
                start = time.time()
                disc_objs = []
                for cont_opt in cont_opts:
                    disc_objs.append(
                        formula.generate_discrete_objective_function(cont_opt)
                    )
                disc_opts_idx = self.discrete_optimizer.optimize(
                    disc_objs, formula.disc_coeffs
                )
                disc_opts_cont_init = [(x[0], cont_opts[x[1]]) for x in disc_opts_idx]
                disc_time_hist.append(time.time() - start)
            if len(formula.cont_coeffs) == 0:
                # Only discrete coefficients
                cont_time_hist.append(0)
                coeff_opt_hist.append([])
                for disc_opt, _ in disc_opts_cont_init:
                    coeff_opt_hist[-1].append(disc_opt)
                break
            #########################
            # Continuous optimization
            #########################
            start = time.time()
            coeff_opt_hist.append([])
            cont_error_hist.append([])
            cont_opts.clear()
            fval_disc_cont_opts.clear()
            for disc_opt, cont_init in disc_opts_cont_init:
                (
                    cont_obj,
                    cont_obj_prime,
                    cont_obj_hessian,
                ) = formula.generate_continuous_objective_function(disc_opt)
                if isinstance(self.continuous_optimizer, BFGSOptimizer):
                    cont_opt, fopt, mes = self.continuous_optimizer.optimize(
                        cont_obj, formula.cont_coeffs, cont_init, cont_obj_prime
                    )
                elif isinstance(self.continuous_optimizer, BFGSJumpOptimizer):
                    expr = formula.substitute_discrete_coefficients(disc_opt)
                    cont_opt, fopt, mes = self.continuous_optimizer.optimize(
                        cont_obj,
                        expr,
                        formula.cont_coeffs,
                        formula.variables,
                        cont_init,
                        cont_obj_prime,
                        data,
                    )
                elif isinstance(self.continuous_optimizer, LMOptimizer):
                    expr = formula.substitute_discrete_coefficients(disc_opt)
                    cont_opt, fopt, mes = self.continuous_optimizer.optimize(
                        data, expr, formula.cont_coeffs, formula.variables, cont_init
                    )
                elif isinstance(self.continuous_optimizer, LMJumpOptimizer):
                    expr = formula.substitute_discrete_coefficients(disc_opt)
                    cont_opt, fopt, mes = self.continuous_optimizer.optimize(
                        cont_obj,
                        expr,
                        formula.cont_coeffs,
                        formula.variables,
                        cont_init,
                        data,
                    )
                if fopt is not None:
                    # converged
                    cont_opts.append(cont_opt)
                    fval_disc_cont_opts.append((fopt, disc_opt, cont_opt))
                    coeff_opt_hist[-1].append(self._merge(disc_opt, cont_opt, formula))
                else:
                    # failed
                    cont_error_hist[-1].append(mes)
            cont_time_hist.append(time.time() - start)
            # Fail in continuous optimization
            if len(cont_opts) == 0:
                if init_method == "normal":
                    cont_opts.append(
                        [rs.normal(0, 1) for _ in range(len(formula.cont_coeffs))]
                    )
                elif init_method == "uniform":
                    cont_opts.append(
                        [rs.uniform(0, 1) for _ in range(len(formula.cont_coeffs))]
                    )
                elif init_method == "order":
                    cont_opt = []
                    for i in range(len(formula.coeffs)):
                        if "c" + str(i) in formula.cont_coeffs_name:
                            cont_opt.append(10 ** rs.uniform(-0.5, 0.5) * coeff_init[i])
                    cont_opts.append(cont_opt)
                else:
                    raise ValueError("Invalid init_method")
            if len(formula.disc_coeffs) == 0:
                # Only continuous coefficients
                break

        # Return the best coefficients
        if len(formula.cont_coeffs) == 0:
            # Only discrete coefficients
            status = "succeeded"
            coeff_opt = disc_opts_cont_init[-1][0]
            ffinal = formula.obj_sympy.subs(
                [(c, v) for c, v in zip(formula.disc_coeffs, coeff_opt)]
            )
        else:
            if len(fval_disc_cont_opts) > 0:
                # Contain continuous coefficients and succeeded in continuous optimization
                status = "succeeded"
                fval_disc_cont_opts.sort()
                disc_opt = fval_disc_cont_opts[0][1]
                cont_opt = fval_disc_cont_opts[0][2]
            else:
                # Contain continuous coefficients but failed in continuous optimization
                status = "failed"
                disc_opt = disc_opts_cont_init[-1][0]
                cont_opt = disc_opts_cont_init[-1][1]
            coeff_opt = self._merge(disc_opt, cont_opt, formula)
            ffinal = formula.obj_sympy.subs(
                [(c, v) for c, v in zip(formula.coeffs, coeff_opt)]
            )

        # return
        if allinfo:
            return (
                formula.formula_simplified.subs(
                    [
                        (c, v)
                        for c, v in zip(
                            ["c" + str(i) for i in range(len(coeff_opt))], coeff_opt
                        )
                    ]
                ),
                status,
                ffinal,
                coeff_opt,
                disc_time_hist,
                cont_time_hist,
                formula.expo_coeffs_name,
                formula.other_coeffs_name,
                formula.disc_coeffs_name,
                formula.cont_coeffs_name,
                coeff_opt_hist,
                cont_error_hist,
            )
        else:
            return (
                formula.formula_simplified.subs(
                    [
                        (c, v)
                        for c, v in zip(
                            ["c" + str(i) for i in range(len(coeff_opt))], coeff_opt
                        )
                    ]
                ),
                status,
            )
