"""class Formula.

This class is used to preprocess the formula and generate objective functions.
"""

from typing import Optional

import sympy


class Formula:
    """class for formula.

    Attributes:
        formula (sympy.core.expr.Expr): original formula
        variables (list[sympy.core.symbol.Symbol]): list of variables in formula
        expo_coeffs (list[sympy.core.symbol.Symbol]): list of exponent coefficients
        other_coeffs (list[sympy.core.symbol.Symbol]): list of other (non-exponential) coefficients
        expo_coeffs_name (list[str]): list of names of exponent coefficients
        other_coeffs_name (list[str]): list of names of other coefficients
        coeffs (list[sympy.core.symbol.Symbol]): list of coefficients
        formula_simplified (sympy.core.expr.Expr): formula to be used to calculate an objective function
        obj_sympy (sympy.core.expr.Expr): objective function
        disc_coeffs (list[sympy.core.symbol.Symbol]): list of discrete coefficients
        cont_coeffs (list[sympy.core.symbol.Symbol]): list of continuous coefficients
        disc_coeffs_name (list[str]): list of names of discrete coefficients
        cont_coeffs_name (list[str]): list of names of continuous coefficients

    Note:
        |- __init__
        |   |- _validate_formula
        |   |- _listup_variable
        |   |- _listup_coefficient
        |   |- _simplify
        |       |- _remove_redundant_numbers
        |- set_objective_function
        |- set_disc_cont_coeffs
        |- generate_discrete_objective_function
        |- generate_continuous_objective_function
    """

    def __init__(self, formula: str | sympy.core.expr.Expr) -> None:
        """intialization

        Args:
            formula (str | sympy.core.expr.Expr): skelton of formula

        Raises:
            ValueError: if coefficients appear both as exponent and as other coefficients

        Examples:
            >>> formula = Formula("x0 * x1 ** c0 + 2 * pi * sin(c1 + x2)")
            >>> formula.formula
            x0*x1**c0 + 2 * pi * sin(c1 + x2)
            >>> formula.variables
            ['x0', 'x1', 'x2']
            >>> formula.expo_coeffs
            ['c0']
            >>> formula.other_coeffs
            ['c1']
            >>> formula.formula_simplified
            x0*x1**c0 + 6.28318530717959*sin(c1 + x2)
        """
        if type(formula) is str:
            formula_complex = sympy.sympify(formula)
        else:
            formula_complex = formula
        # change symbols to real-valued ones
        symbols = sympy.symbols(
            [s.name for s in formula_complex.free_symbols], real=True
        )
        self.formula = formula_complex.subs(
            list(zip(formula_complex.free_symbols, symbols))
        )
        self._validate_formula(self.formula)
        self.variables: list[sympy.core.symbol.Symbol] = sorted(
            self._listup_variable(self.formula), key=lambda s: s.name
        )
        expo_coeffs, other_coeffs = self._listup_coefficient(self.formula, None, None)
        self.expo_coeffs: list[sympy.core.symbol.Symbol] = sorted(
            expo_coeffs, key=lambda s: s.name
        )
        self.other_coeffs: list[sympy.core.symbol.Symbol] = sorted(
            other_coeffs, key=lambda s: s.name
        )
        self.expo_coeffs_name: list[str] = [s.name for s in self.expo_coeffs]
        self.other_coeffs_name: list[str] = [s.name for s in self.other_coeffs]
        self.coeffs: list[sympy.core.symbol.Symbol] = []
        eidx: int = 0
        oidx: int = 0
        for i in range(len(self.expo_coeffs) + len(self.other_coeffs)):
            if f"c{i}" in [s.name for s in self.expo_coeffs]:
                self.coeffs.append(self.expo_coeffs[eidx])
                eidx += 1
            else:
                self.coeffs.append(self.other_coeffs[oidx])
                oidx += 1
        if len(set(expo_coeffs) & set(other_coeffs)) > 0:
            raise ValueError(
                f"Coefficients {set(expo_coeffs) & set(other_coeffs)} appear both as exponent and as other coefficients"
            )
        self.formula_simplified: sympy.core.expr.Expr = self._simplify(self.formula)

    def _validate_formula(self, formula: sympy.core.expr.Expr) -> None:
        """output error if formula does not have a predefined type.

        Args:
            formula (sympy.core.expr.Expr): formula to be validated

        Raises:
            TypeError: if formula does not have a predefined type
        """
        for arg in formula.args:
            self._validate_formula(arg)  # type: ignore
        if type(self.formula) in [
            sympy.core.symbol.Symbol,
            sympy.core.add.Add,
            sympy.core.mul.Mul,
            sympy.core.power.Pow,
            sympy.functions.elementary.exponential.log,
            sympy.functions.elementary.trigonometric.sin,
            sympy.functions.elementary.trigonometric.cos,
            sympy.functions.elementary.hyperbolic.tanh,
            sympy.functions.elementary.exponential.exp,
        ]:
            pass
        else:
            raise TypeError(
                f"Invalid type of formula: {type(self.formula)}\n"
                f"formula: {self.formula}"
            )

    def _listup_variable(
        self, formula: sympy.core.expr.Expr
    ) -> list[sympy.core.symbol.Symbol]:
        """return list of variables (["x0", "x1", ...]) in formula.

        Args:
            formula (sympy.core.expr.Expr): target formula

        Returns:
            list[sympy.core.symbol.Symbol]: list of variables in formula
        """
        if type(formula) is sympy.core.symbol.Symbol and formula.name[0] == "x":
            return [formula]
        elif formula.is_number:
            return []
        variables: list[sympy.core.symbol.Symbol] = []
        for arg in formula.args:
            variables += self._listup_variable(arg)  # type: ignore
        return list(set(variables))

    def _listup_coefficient(
        self,
        formula: sympy.core.expr.Expr,
        parent_type: Optional[type],
        arg_pos: Optional[int],
    ) -> tuple[list[sympy.core.symbol.Symbol], list[sympy.core.symbol.Symbol]]:
        """return list of exponent coefficients and list of other coefficients in formula

        Args:
            formula (sympy.core.expr.Expr): target formula
            parent_type (Optional[type]): type of parent node
            arg_pos (Optional[int]): position of formula in parent node

        Returns:
            tuple[list[sympy.core.symbol.Symbol], list[sympy.core.symbol.Symbol]]:
            list of exponent coefficients and list of other coefficients in formula
        """
        if parent_type == sympy.core.power.Pow and arg_pos == 1:
            if type(formula) is sympy.core.symbol.Symbol and formula.name[0] == "c":
                return [formula], []
            elif (
                type(formula) is sympy.core.mul.Mul
                or type(formula) is sympy.core.add.Add
            ):
                args = formula.args
                symbol = []
                number = []
                for arg in args:
                    if arg.is_number:
                        number.append(arg)
                    elif type(arg) is sympy.core.symbol.Symbol and arg.name[0] == "c":
                        symbol.append(arg)
                if len(symbol) == 1 and len(number) == len(args) - 1:
                    return symbol, []
        elif type(formula) is sympy.core.symbol.Symbol:
            if formula.name[0] == "c":
                return [], [formula]
            else:
                return [], []
        elif formula.is_number:
            return [], []
        expo_coeffs = []
        other_coeffs = []
        for arg_pos, arg in enumerate(formula.args):
            expo, other = self._listup_coefficient(arg, type(formula), arg_pos)
            expo_coeffs += expo
            other_coeffs += other
        return list(set(expo_coeffs)), list(set(other_coeffs))

    def _simplify(self, formula: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
        """replace pi with its numerical value and remove redundant coefficients.

        Args:
            formula (sympy.core.expr.Expr): formula to be simplified

        Returns:
            sympy.core.expr.Expr: simplified formula
        """
        formula_tmp = formula.evalf(subs={"pi": sympy.pi})
        return self._remove_redundant_numbers(formula_tmp)

    def _remove_redundant_numbers(
        self, formula: sympy.core.expr.Expr
    ) -> sympy.core.expr.Expr:
        """remove redundant numbers

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
            # Only Add and Mul can have multiple numbers as numbers
            numbers = []
            non_numbers = []
            for arg in formula.args:
                if arg.is_number:
                    numbers.append(arg)
                else:
                    non_numbers.append(self._remove_redundant_numbers(arg))  # type: ignore
            if len(numbers) == 0:
                return formula.func(*non_numbers)
            else:
                new_number = formula.func(*numbers).evalf()
                return formula.func(new_number, *non_numbers)
        else:
            return formula.func(
                *[self._remove_redundant_numbers(arg) for arg in formula.args]  # type: ignore
            )

    def set_objective_function(
        self,
        data: list[list[float]],
    ) -> None:
        """prepare objective function

        Args:
            data (list[list[float]]): from first to the second last columns are variables in formula (["x0", "x1", ...]), and the last column is target("y")

        Raises:
            ValueError: if names or the number of variables in formula don't match data

        Examples:
            >>> formula = Formula("c0 * x0 + c1 + x1")
            >>> formula.set_objective_function([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> formula.obj_sympy
            8.0*(c0 + 0.25*c1 - 0.25)**2 + (1.0*c0 + c1 - 1.0)**2/2
        """
        # Check vairables in formula
        if [s.name for s in self.variables] != [
            "x" + str(i) for i in range(len(data[0]) - 1)
        ]:
            raise ValueError(
                "Names or the number of variables in formula don't match data"
            )

        # Substitute variables with data
        obj_sympy: sympy.core.expr.Expr = sympy.sympify(0)
        for row in data:
            obj_sympy += (
                self.formula_simplified.subs(
                    [(self.variables[i], row[i]) for i in range(len(self.variables))]
                )
                - row[-1]
            ) ** 2
        obj_sympy /= len(data)

        # Keep in instance variables
        self.obj_sympy: sympy.core.expr.Expr = obj_sympy

    def set_disc_cont_coeffs(
        self,
        discrete_optimizer: bool,
    ) -> None:
        """set discrete and continuous coefficients

        Args:
            discrete_optimizer (bool): whether discrete optimizer is used or not

        Examples:
            >>> formula = Formula("x0 * x1 ** c0 + 2 * pi * sin(c1 + x2)")
            >>> formula.set_disc_cont_coeffs(True)
            >>> formula.disc_coeffs
            ['c0']
            >>> formula.cont_coeffs
            ['c1']
        """
        if discrete_optimizer:
            self.disc_coeffs: list[sympy.core.symbol.Symbol] = self.expo_coeffs
            self.cont_coeffs: list[sympy.core.symbol.Symbol] = self.other_coeffs
        else:
            self.disc_coeffs = []
            self.cont_coeffs = sorted(
                self.expo_coeffs + self.other_coeffs, key=lambda s: s.name
            )
        self.disc_coeffs_name = [s.name for s in self.disc_coeffs]
        self.cont_coeffs_name = [s.name for s in self.cont_coeffs]

    def generate_discrete_objective_function(
        self, cont_coeff_val: list[float]
    ) -> sympy.core.expr.Expr:
        """subsitute continuous coefficients and return objective function for discrete optimization

        Args:
            cont_coeff_val (list[float]): values of continuous coefficients

        Returns:
            sympy.core.expr.Expr: objective function for discrete optimization

        Raises:
            AssertionError: if the number of continuous coefficients does not match the number of values
        """
        assert len(self.cont_coeffs) == len(
            cont_coeff_val
        ), f"The number of continuous coefficients {len(self.cont_coeffs)} does not match the number of values {len(cont_coeff_val)}"
        return self.obj_sympy.subs(
            [(c, v) for c, v in zip(self.cont_coeffs, cont_coeff_val)]
        )

    def generate_continuous_objective_function(
        self, disc_coeff_val: list[float]
    ) -> tuple[
        sympy.core.expr.Expr,
        list[sympy.core.expr.Expr],
        list[list[sympy.core.expr.Expr]],
    ]:
        """subsitute discrete coefficients and return objective function for continuous optimization, its gradient, and its hessian

        Args:
            disc_coeff_val (list[float]): values of discrete coefficients

        Returns:
            tuple[ sympy.core.expr.Expr, list[sympy.core.expr.Expr], list[list[sympy.core.expr.Expr]], ]
            : objective function for continuous optimization, its gradient, and its hessian
        """
        assert len(self.disc_coeffs) == len(
            disc_coeff_val
        ), f"The number of discrete coefficients {len(self.disc_coeffs)} does not match the number of values {len(disc_coeff_val)}"
        cont_obj = self.obj_sympy.subs(
            [(c, v) for c, v in zip(self.disc_coeffs, disc_coeff_val)]
        )
        cont_obj_prime = [sympy.diff(cont_obj, c) for c in self.cont_coeffs]
        cont_obj_hessian = [
            [sympy.diff(cont_obj_prime[i], c) for c in self.cont_coeffs]
            for i in range(len(self.cont_coeffs))
        ]
        return cont_obj, cont_obj_prime, cont_obj_hessian

    def substitute_discrete_coefficients(
        self, disc_coeff_val: list[float]
    ) -> sympy.core.expr.Expr:
        """substitute discrete coefficients and return formula for continuous optimization

        Args:
            disc_coeff_val (list[float]): values of discrete coefficients

        Returns:
            sympy.core.expr.Expr: formula.formula_simplified with discrete coefficients substituted
        """
        assert len(self.disc_coeffs) == len(
            disc_coeff_val
        ), f"The number of discrete coefficients {len(self.disc_coeffs)} does not match the number of values {len(disc_coeff_val)}"
        return self.formula_simplified.subs(
            [(c, v) for c, v in zip(self.disc_coeffs, disc_coeff_val)]
        )
