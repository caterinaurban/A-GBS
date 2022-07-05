"""
Symbolic Constant Propagation (Variant 3)
=========================================

:Authors: Caterina Urban
"""
import time
from copy import deepcopy
from math import inf, isclose
from typing import Set, List, Dict, Tuple

from apronpy.manager import PyManager
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprDiscr, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from pip._vendor.colorama import Fore, Style
from pulp import pulp, PULP_CBC_CMD

from agbs.abstract_domains.interval_domain import IntervalLattice
from agbs.abstract_domains.state import State
from agbs.core.expressions import VariableIdentifier, Expression, BinaryBooleanOperation, \
    BinaryComparisonOperation, Literal, UnaryArithmeticOperation
from agbs.core.utils import copy_docstring


rtype = TexprRtype.AP_RTYPE_REAL
rdir = TexprRdir.AP_RDIR_RND


def texpr_to_dict(texpr):

    def do(texpr0, env):
        if texpr0.discr == TexprDiscr.AP_TEXPR_CST:
            result = dict()
            t0 = '{}'.format(texpr0.val.cst)
            t1 = eval(t0)
            t2 = str(t1)
            t3 = float(t2)
            result['_'] = t3
            return result
        elif texpr0.discr == TexprDiscr.AP_TEXPR_DIM:
            result = dict()
            result['{}'.format(env.var_of_dim[texpr0.val.dim.value].decode('utf-8'))] = 1.0
            return result
        else:
            assert texpr0.discr == TexprDiscr.AP_TEXPR_NODE
            left = do(texpr0.val.node.contents.exprA.contents, env)
            op = texpr0.val.node.contents.op
            if texpr0.val.node.contents.exprB:
                right = do(texpr0.val.node.contents.exprB.contents, env)
            if op == TexprOp.AP_TEXPR_ADD:
                result = deepcopy(left)
                for var, val in right.items():
                    if var in result:
                        result[var] += val
                    else:
                        result[var] = val
                return result
            elif op == TexprOp.AP_TEXPR_SUB:
                result = deepcopy(left)
                for var, val in right.items():
                    if var in result:
                        result[var] -= val
                    else:
                        result[var] = -val
                return result
            elif op == TexprOp.AP_TEXPR_MUL:
                # print('multiplying')
                # print('left: ', left)
                # print('right: ', right)
                result = dict()
                if '_' in left and len(left) == 1:
                    for var, val in right.items():
                        result[var] = left['_'] * right[var]
                elif '_' in right and len(right) == 1:
                    for var, val in left.items():
                        result[var] = right['_'] * left[var]
                else:
                    assert False
                # print('result: ', result)
            elif op == TexprOp.AP_TEXPR_NEG:
                result = deepcopy(left)
                for var, val in result.items():
                    result[var] = -val
        return result

    texpr1 = texpr.texpr1.contents
    return do(texpr1.texpr0.contents, texpr1.env.contents)


def evaluate(dictionary, bounds):
    result = IntervalLattice(0, 0)
    for var, val in dictionary.items():
        coeff = IntervalLattice(val, val)
        if var != '_':
            result = result._add(coeff._mult(bounds[var]))
        else:
            result = result._add(coeff)
    return result


def evaluate_with_constraints(dictionary, bounds, constraints):
    ranges = {"__dummy": (0, 0)}
    for var, val in bounds.items():
        ranges[var] = (val.lower, val.upper)
    current = dict(
        objective=dict(
            name='OBJ',
            coefficients=[
                {"name": name, "value": value} for name, value in dictionary.items() if name != '_'
            ]
        ),
        constraints=[
            dict(
                sense=status,
                pi=None,
                constant=expression['_'],
                name=None,
                coefficients=[
                    {"name": name, "value": value} for name, value in expression.items() if name != '_'
                ],
            ) for expression, status in constraints
        ],
        variables=[
            dict(lowBound=l, upBound=u, cat="Continuous", varValue=None, dj=None, name=v)
            for v, (l, u) in ranges.items()],
        parameters=dict(name="NoName", sense=1, status=0, sol_status=0),
        sos1=list(),
        sos2=list(),
    )
    _current = deepcopy(current)
    _current['parameters'] = {'name': '', 'sense': 1, 'status': 1, 'sol_status': 1}  # min
    _, _problem = pulp.LpProblem.fromDict(_current)
    _problem.solve(PULP_CBC_CMD(msg=False))
    lower = pulp.value(_problem.objective) + dictionary['_']
    current_ = deepcopy(current)
    current_['parameters'] = {'name': '', 'sense': -1, 'status': 1, 'sol_status': 1}  # max
    _, problem_ = pulp.LpProblem.fromDict(current_)
    problem_.solve(PULP_CBC_CMD(msg=False))
    upper = pulp.value(problem_.objective) + dictionary['_']
    return IntervalLattice(lower, upper)


def substitute_in_dict(todict, var, rhs):
    result = todict
    key = str(var)
    coeff = result.get(key, 0)
    result.pop(key, None)
    for var, val in rhs.items():
        if var in result:
            result[var] += coeff * val
        else:
            result[var] = coeff * val
    return result


class SymbolicState(State):
    """Interval+Symbolic (Variant 3)

    .. document private methods
    .. automethod:: Symbolic3State._assign
    .. automethod:: Symbolic3State._assume
    .. automethod:: Symbolic3State._output
    .. automethod:: Symbolic3State._substitute

    """
    def __init__(self, inputs: Set[VariableIdentifier], precursory: State = None):
        super().__init__(precursory=precursory)
        self.inputs = {input.name for input in inputs}
        self.bounds: Dict[str, IntervalLattice] = dict()
        for input in self.inputs:
            self.bounds[input] = IntervalLattice(0, 1)
        self.symbols: Dict[str, Tuple[str, dict]] = dict()
        self.expressions = dict()
        self.polarities = dict()
        self.ranges = dict()
        self.flag = None

    @copy_docstring(State.bottom)
    def bottom(self):
        for var in self.bounds:
            self.bounds[var].bottom()
        return self

    @copy_docstring(State.top)
    def top(self):
        for var in self.bounds:
            self.bounds[var].top()
        return self

    def __repr__(self):
        items = sorted(self.bounds.items(), key=lambda x: x[0])
        return ", ".join("{} -> {}".format(variable, value) for variable, value in items)

    @copy_docstring(State.is_bottom)
    def is_bottom(self) -> bool:
        return any(element.is_bottom() for element in self.bounds.values())

    @copy_docstring(State.is_top)
    def is_top(self) -> bool:
        return all(element.is_top() for element in self.bounds.values())

    @copy_docstring(State._less_equal)
    def _less_equal(self, other: 'SymbolicState') -> bool:
        raise NotImplementedError(f"Call to _is_less_equal is unexpected!")

    @copy_docstring(State._join)
    def _join(self, other: 'SymbolicState') -> 'SymbolicState':
        for var in self.bounds:
            self.bounds[var].join(other.bounds[var])
        return self

    @copy_docstring(State._meet)
    def _meet(self, other: 'SymbolicState') -> 'SymbolicState':
        for var in self.bounds:
            self.bounds[var].meet(other.bounds[var])
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'SymbolicState') -> 'SymbolicState':
        raise NotImplementedError(f"Call to _widening is unexpected!")

    @copy_docstring(State._assign)
    def _assign(self, left: Expression, right: Expression) -> 'SymbolicState':
        raise NotImplementedError(f"Call to _assign is unexpected!")

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, bwd: bool = False) -> 'SymbolicState':
        raise NotImplementedError(f"Call to _assume is unexpected!")

    def assume(self, condition, manager: PyManager = None, bwd: bool = False) -> 'SymbolicState':
        if self.is_bottom():
            return self
        if isinstance(condition, tuple):
            condition = list(condition)
        if isinstance(condition, list):
            for feature, (lower, upper) in condition:
                self.bounds[feature.name].meet(IntervalLattice(lower, upper))
            return self
        elif isinstance(condition, BinaryBooleanOperation):
            if condition.operator == BinaryBooleanOperation.Operator.Or:
                right = deepcopy(self).assume(condition.right, bwd=bwd)
                return self.assume(condition.left, bwd=bwd).join(right)
            elif condition.operator == BinaryBooleanOperation.Operator.And:
                assert isinstance(condition.left, BinaryComparisonOperation)
                assert isinstance(condition.right, BinaryComparisonOperation)
                if isinstance(condition.left.left, Literal):
                    lower = eval(condition.left.left.val)
                else:
                    assert isinstance(condition.left.left, UnaryArithmeticOperation)
                    assert condition.left.left.operator == UnaryArithmeticOperation.Operator.Sub
                    assert isinstance(condition.left.left.expression, Literal)
                    lower = -eval(condition.left.left.expression.val)
                assert condition.left.operator == BinaryComparisonOperation.Operator.LtE
                assert isinstance(condition.left.right, VariableIdentifier)
                assert isinstance(condition.right.left, VariableIdentifier)
                assert condition.right.operator == BinaryComparisonOperation.Operator.LtE
                if isinstance(condition.right.right, Literal):
                    upper = eval(condition.right.right.val)
                else:
                    assert isinstance(condition.right.right, UnaryArithmeticOperation)
                    assert condition.right.right.operator == UnaryArithmeticOperation.Operator.Sub
                    assert isinstance(condition.right.right.expression, Literal)
                    upper = -eval(condition.right.right.expression.val)
                assert condition.left.right.name == condition.right.left.name
                self.bounds[condition.left.right.name].meet(IntervalLattice(lower, upper))
                return self
        elif isinstance(condition, BinaryComparisonOperation):
            if condition.operator == BinaryComparisonOperation.Operator.Gt:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = eval(condition.right.val)
                upper = inf
                self.bounds[condition.left.name].meet(IntervalLattice(lower, upper))
                return self
            elif condition.operator == BinaryComparisonOperation.Operator.LtE:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = -inf
                upper = eval(condition.right.val)
                self.bounds[condition.left.name].meet(IntervalLattice(lower, upper))
                return self
        # elif isinstance(condition, PyTcons1):
        #     abstract1 = self.domain(manager, self.environment, array=PyTcons1Array([condition]))
        #     self.state = self.state.meet(abstract1)
        #     return self
        elif isinstance(condition, set):
            assert len(condition) == 1
            self.assume(condition.pop(), bwd=bwd)
            return self
        raise NotImplementedError(f"Assumption of {condition.__class__.__name__} is unsupported!")

    @copy_docstring(State._substitute)
    def _substitute(self, left: Expression, right: Expression) -> 'SymbolicState':
        raise NotImplementedError(f"Call to _substitute is unexpected!")

    def affine(self, left: List[PyVar], right: List[PyTexpr1], constraints=None) -> 'SymbolicState':
        if self.is_bottom():
            return self
        assignments = dict()
        for lhs, expr in zip(left, right):
            name = str(lhs)
            rhs = texpr_to_dict(expr)
            for sym, val in self.symbols.values():
                rhs = substitute_in_dict(rhs, sym, val)
            assignments[name] = (name, rhs)
            if constraints:
                bound = evaluate_with_constraints(rhs, self.bounds, constraints)
            else:
                bound = evaluate(rhs, self.bounds)
            self.bounds[name] = IntervalLattice(bound.lower, bound.upper)
            if bound.lower < 0 and 0 < bound.upper:
                self.expressions[name] = rhs
                self.polarities[name] = (bound.lower + bound.upper) / (bound.upper - bound.lower)
                self.ranges[name] = bound.upper - bound.lower
        self.symbols = assignments
        return self

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'SymbolicState':
        if self.is_bottom():
            return self
        self.flag = None

        name = str(stmt)
        lattice: IntervalLattice = self.bounds[name]
        lower, upper = lattice.lower, lattice.upper
        if upper <= 0 or inactive:
            zero = dict()
            zero['_'] = 0.0
            self.symbols[name] = (name, zero)
            self.bounds[name] = IntervalLattice(0, 0)
            self.flag = -1
        elif 0 <= lower or active:
            if active and lower < 0:
                bounds = self.bounds[name]
                self.bounds[name] = bounds.meet(IntervalLattice(0, upper))
                del self.symbols[name]
            self.flag = 1
        else:
            _active, _inactive = deepcopy(self.bounds), deepcopy(self.bounds)
            _active[name] = _active[name].meet(IntervalLattice(0, upper))
            _inactive[name] = _inactive[name].meet(IntervalLattice(0, 0))

            if any(element.is_bottom() for element in _active.values()):
                zero = dict()
                zero['_'] = 0.0
                self.symbols[name] = (name, zero)
                self.flag = -1
            elif any(element.is_bottom() for element in _inactive.values()):
                self.flag = 1
            else:
                del self.symbols[name]
                self.flag = None

            join = dict()
            for variable, itv in _active.items():
                join[variable] = itv.join(_inactive[variable])
            self.bounds[name] = join[name].meet(IntervalLattice(0, upper))
            self.flag = None

        return self

    def outcome(self, outcomes: Set[VariableIdentifier]):
        found = None
        if self.is_bottom():
            found = '⊥'
        else:
            for chosen in outcomes:
                outcome = self.bounds[chosen.name]
                lower = outcome.lower
                unique = True
                remaining = outcomes - {chosen}
                for discarded in remaining:
                    alternative = self.bounds[discarded.name]
                    upper = alternative.upper
                    if lower <= upper:
                        unique = False
                        break
                if unique:
                    found = chosen
                    break
        return found

    def get_bounds(self, var_name):
        return self.bounds[var_name]

    def resize_bounds(self, var_name, new_bounds):
        self.bounds[var_name] = IntervalLattice(new_bounds.lower, new_bounds.upper)

    def get_expressions(self, var_name):
        return self.expressions[var_name]

    def log(self, output_names, full=True):
        """log of the state bounds (usually only Input/Output) of the state after a forward analysis step

        :param state: state of the analsis after a forward application
        :param outputs: set of outputs name
        :param full: True for full print or False for just Input/Output (Default False)
        """
        input_color = Fore.YELLOW
        output_color = Fore.MAGENTA
        mid_color = Fore.LIGHTBLACK_EX
        error_color = Fore.RED
        output_names = {str(k) for k in output_names}

        print("Forward Analysis (", Style.RESET_ALL, end='', sep='')
        print(input_color + "Input", Style.RESET_ALL, end='', sep='')
        print("|", Style.RESET_ALL, end='', sep='')
        if full:
            print(mid_color + "Hidden", Style.RESET_ALL, end='', sep='')
            print("|", Style.RESET_ALL, end='', sep='')

        print(output_color + "Output", Style.RESET_ALL, end='', sep='')
        print("): {", Style.RESET_ALL)

        if hasattr(self, "bounds") and isinstance(self.bounds, dict):
            inputs = [f"  {k} -> {self.bounds[k]}" for k in self.inputs]
            inputs.sort()
            print(input_color + "\n".join(inputs), Style.RESET_ALL)
            if full:
                mid_states = [f"  {k} -> {self.bounds[k]} | {self.symbols.get(k, {})}" for k in self.bounds.keys() - self.inputs - output_names]
                mid_states.sort()
                print(mid_color + "\n".join(mid_states), Style.RESET_ALL)
            outputs = [f"  {k} -> {self.bounds[k]} | {self.symbols.get(k, {})}" for k in output_names]
            outputs.sort()
            print(output_color + "\n".join(outputs), Style.RESET_ALL)
        else:
            print(error_color + "Unable to show bounds on the param 'state'" +
                "\n  > missing attribute 'state.bounds', or 'state.bounds' is not a dictionary" +
                "\n  > next state logs will be hidden", Style.RESET_ALL)
            self._log = True

        print("}\n", Style.RESET_ALL)
