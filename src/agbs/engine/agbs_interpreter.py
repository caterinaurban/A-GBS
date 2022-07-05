from copy import deepcopy
from queue import Queue
from typing import Set, Dict, Tuple, List, FrozenSet

from pulp import pulp, PULP_CBC_CMD
from pip._vendor.colorama import Fore, Style

from agbs.abstract_domains.state import State
from agbs.abstract_domains.symbolic_domain import substitute_in_dict, evaluate_with_constraints
from agbs.core.cfg import Node, Function, Activation
from agbs.core.expressions import VariableIdentifier
from agbs.core.statements import Call
from agbs.semantics.forward import DefaultForwardSemantics


class AGBSInterpreter:

    def __init__(self, cfg, domain, semantics, nodes):
        self._cfg = cfg
        self._domain = domain
        self._semantics = semantics
        self._inputs = nodes[0]
        self._relus_name2node = nodes[1]
        self._relus_node2name = {n: v for v, n in nodes[1].items()}
        self._outputs = nodes[2]
        self._initial = None
        self._pattern = None

    @property
    def cfg(self):
        return self._cfg

    @property
    def domain(self):
        return self._domain

    @property
    def semantics(self):
        return self._semantics

    @property
    def inputs(self) -> Set[VariableIdentifier]:
        return self._inputs

    @property
    def relus_name2node(self) -> Dict[VariableIdentifier, Node]:
        return self._relus_name2node

    @property
    def relus_node2name(self) -> Dict[Node, VariableIdentifier]:
        return self._relus_node2name

    @property
    def outputs(self) -> Set[VariableIdentifier]:
        return self._outputs

    @property
    def initial(self):
        return deepcopy(self._initial)

    @property
    def pattern(self) -> List[Tuple[Set[str], Set[str]]]:
        return self._pattern

    def fwd(self, initial, forced_active=None, forced_inactive=None, constraints=None):
        """Single run of the forward analysis with the abstract domain"""
        worklist = Queue()
        worklist.put(self.cfg.in_node)
        state = deepcopy(initial)
        activations = list()
        activated, deactivated, uncertain = set(), set(), set()
        while not worklist.empty():
            current: Node = worklist.get()  # retrieve the current node
            if isinstance(current, Function):
                if activated or deactivated or uncertain:
                    activations.append((frozenset(activated), frozenset(deactivated), frozenset(uncertain)))
                    activated, deactivated, uncertain = set(), set(), set()
                state = state.affine(current.stmts[0], current.stmts[1], constraints=constraints)
            elif isinstance(current, Activation):
                if forced_active and current in forced_active:
                    state = state.relu(current.stmts, active=True)
                    activated.add(current)
                elif forced_inactive and current in forced_inactive:
                    state = state.relu(current.stmts, inactive=True)
                    deactivated.add(current)
                else:
                    state = state.relu(current.stmts)
                    if state.is_bottom():
                        deactivated.add(current)
                    if state.flag:
                        if state.flag > 0:
                            activated.add(current)
                        else:
                            deactivated.add(current)
                    if current not in activated and current not in deactivated:
                        uncertain.add(current)
            else:
                if activated or deactivated or uncertain:
                    activations.append((frozenset(activated), frozenset(deactivated), frozenset(uncertain)))
                    activated, deactivated, uncertain = set(), set(), set()
                for stmt in reversed(current.stmts):
                    state = self.semantics.assume_call_semantics(stmt, state)
            # update worklist
            for node in self.cfg.successors(current):
                worklist.put(self.cfg.nodes[node.identifier])

        # state.log(self.outputs)

        # get lower-bounds and upper-bounds for all outputs
        lowers: Dict[str, float] = dict((o.name, state.bounds[o.name].lower) for o in self.outputs)
        uppers: Dict[str, float] = dict((o.name, state.bounds[o.name].upper) for o in self.outputs)

        self.print_pattern(activations)
        if all(lower >= 0 for lower in lowers.values()):
            return 1, activations, None
        elif any(upper < 0 for upper in uppers.values()):
            return -1, activations, None
        else:
            # retrieve uncertain relus
            uncertain: List[Set[str]] = list()
            for pack in activations:
                if pack[2]:
                    uncertain.append({self.relus_node2name[n].name for n in pack[2]})
            if len(uncertain) == 0:
                # for o in self.outputs:
                #     if o.name in state.expressions:
                #         # back-substitution
                #         current = state.expressions[o.name]
                #         while any(variable in state.expressions for variable in state.expressions[o.name].keys()):
                #             for sym, val in state.expressions.items():
                #                 if sym in current:
                #                     current = substitute_in_dict(current, sym, val)
                #             state.expressions[o.name] = current
                #         # evaluate_with_constraints(state.expressions[o.name])
                return None, activations, None
            # pick output with smallest lower-bound
            lowers: Dict[str, float] = dict((o.name, state.bounds[o.name].lower) for o in self.outputs)
            picked: str = min(lowers, key=lowers.get)
            # retrieve its symbolic expression
            expression: Dict[str, float] = state.symbols[picked][1]
            # remove inputs and constants from symbolic expression to only remain with relus
            for i in self.inputs:
                expression.pop(i.name, None)
            expression.pop('_')
            # remove forced active relus
            for r in forced_active:
                expression.pop(self.relus_node2name[r].name, None)
            # rank relus by layer
            layer_score = lambda relu_name: len(uncertain) - list(relu_name in u for u in uncertain).index(True)
            # rank relus by coefficient
            coeff_rank = sorted(expression.items(), key=lambda item: (layer_score(item[0]), abs(item[1])), reverse=True)
            coeff_score = lambda relu_name: list(relu_name == item[0] for item in coeff_rank).index(True)
            # rank relu by range size
            range_size = lambda relu_name: state.ranges[relu_name]
            range_rank = sorted(list((relu, range_size(relu)) for relu in expression.keys()), key=lambda item: (layer_score(item[0]), item[1]), reverse=True)
            range_score = lambda relu_name: list(relu_name == item[0] for item in range_rank).index(True)
            # rank relu by polarities
            polarity = lambda relu_name: abs(state.polarities[relu_name])
            pol_rank = sorted(list((relu, polarity(relu)) for relu in expression.keys()), key=lambda item: (len(uncertain) - layer_score(item[0]), item[1]))
            pol_score = lambda relu_name: list(relu_name == item[0] for item in pol_rank).index(True)
            # determine final rank
            rank = lambda relu: coeff_score(relu) + range_score(relu) + pol_score(relu)
            ranked = sorted(list((relu, rank(relu)) for relu in expression.keys()), key=lambda item: item[1])
            # return chosen uncertain relu(s)
            choice: str = ranked[0][0]
            r_layer = list(choice in u for u in uncertain).index(True)
            r_coeff = expression[choice]
            r_range = state.ranges[choice]
            r_polarity = state.polarities[choice]
            r_rank = '(layer: {}, coeff: {}, range: {}, polarity, {})'.format(r_layer, r_coeff, r_range, r_polarity)
            print('Choice: ', choice, r_rank)
            chosen: Set[str] = {choice}
            expression = state.expressions[choice]
            r_chosen = list()
            for name, expr in state.expressions.items():
                if name != choice and expr == expression:
                    chosen.add(name)
                    r_chosen.append(name)
            print('Chosen: {}'.format(', '.join(r_chosen)))
            # back-substitution
            current = dict(expression)
            while any(variable in state.expressions for variable in expression.keys()):
                for sym, val in state.expressions.items():
                    if sym in current:
                        current = substitute_in_dict(current, sym, val)
                expression = current
            return 0, activations, (chosen, expression)

    def is_redundant(self, constraint, ranges, constraints):
        """
        Set the objective coefficients to those of one of the constraints, disable that constraint and solve the LP:
    - if the constraint was a LE maximize. In case the optimal value is less or equal to the rhs, the constraint is redundant
    - analogously if the constraint was GE minimize. disabled constraint is redundant if the optimal value or greater or equal to the rhs.
        """
        (expr, activity) = constraint
        _ranges = dict(ranges)
        current = dict(
            objective=dict(
                name=None,
                coefficients=[
                    {"name": name, "value": value} for name, value in expr.items() if name != '_'
                ]),
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
            variables=[dict(lowBound=l, upBound=u, cat="Continuous", varValue=None, dj=None, name=v) for v, (l, u) in
                       _ranges.items()],
            parameters=dict(name="NoName", sense=-activity, status=0, sol_status=0),
            sos1=list(),
            sos2=list(),
        )
        var, problem = pulp.LpProblem.fromDict(current)
        problem.solve(PULP_CBC_CMD(msg=False))
        if activity < 0:
            return pulp.value(problem.objective) + expr['_'] <= 0
        elif activity > 0:
            return pulp.value(problem.objective) + expr['_'] >= 0

    def to_pulp(self, ranges, constraints):
        _ranges = dict(ranges)
        _ranges["__dummy"] = (0, 0)
        current = dict(
            objective=dict(name=None, coefficients=[{"name": "__dummy", "value": 1}]),
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
            variables=[dict(lowBound=l, upBound=u, cat="Continuous", varValue=None, dj=None, name=v) for v, (l, u) in
                       _ranges.items()],
            parameters=dict(name="NoName", sense=1, status=0, sol_status=0),
            sos1=list(),
            sos2=list(),
        )
        var, problem = pulp.LpProblem.fromDict(current)
        problem.solve(PULP_CBC_CMD(msg=False))
        if problem.status == -1:
            return None
        else:
            bounds = dict()
            for name in ranges:
                _current = deepcopy(current)
                _current['objective'] = {'name': 'OBJ', 'coefficients': [{'name': name, 'value': 1}]}
                _current['parameters'] = {'name': '', 'sense': 1, 'status': 1, 'sol_status': 1}  # min
                _, _problem = pulp.LpProblem.fromDict(_current)
                _problem.solve(PULP_CBC_CMD(msg=False))
                lower = pulp.value(_problem.objective)
                current_ = deepcopy(current)
                current_['objective'] = {'name': 'OBJ', 'coefficients': [{'name': name, 'value': 1}]}
                current_['parameters'] = {'name': '', 'sense': -1, 'status': 1, 'sol_status': 1}  # max
                _, problem_ = pulp.LpProblem.fromDict(current_)
                problem_.solve(PULP_CBC_CMD(msg=False))
                upper = pulp.value(problem_.objective)
                bounds[VariableIdentifier(name)] = (lower, upper)
            return list(bounds.items())

    def original_status(self, relu_names):
        # retrieve uncertain relus
        for pack in self.pattern:
            if all(name in pack[0] for name in relu_names):
                return 1
            elif all(name in pack[1] for name in relu_names):
                return -1
        raise ValueError

    def print_pattern(self, activations):
        activated, deactivated, uncertain = 0, 0, 0
        # print('Activation Pattern: {', end='')
        for (a, d, u) in activations:
            # print('[')
            activated += len(a)
            # print('activated: ', ','.join(self.relus_node2name[n].name for n in a))
            deactivated += len(d)
            # print('deactivated: ', ','.join(self.relus_node2name[n].name for n in d))
            uncertain += len(u)
            # print('uncertain: ', ','.join(self.relus_node2name[n].name for n in u))
            # print(']', end='')
        # print('}')
        print('#Active: ', activated, '#Inactive: ', deactivated, '#Uncertain: ', uncertain, '\n')

    def search(self, in_nxt, in_f_active, in_f_inactive, in_constraints):
        status = 0
        nxt = in_nxt
        f_active, f_inactive = set(in_f_active), set(in_f_inactive)
        constraints = list(in_constraints)
        alternatives = list()
        while status == 0:
            if nxt is None:
                r_cstr = ''
                for i, (e, a) in enumerate(constraints):
                    r_expr = ' + '.join('({})*{}'.format(v, n) for n, v in e.items() if n != '_')
                    r_expr = r_expr + ' + {}'.format(e['_'])
                    if a > 0:
                        r_cstr = r_cstr + '[{}+]: '.format(i) + r_expr + ' >= 0\n'
                    else:
                        r_cstr = r_cstr + '[{}-]: '.format(i) + r_expr + ' <= 0\n'
                print('Constraints: ', r_cstr)
                print('\n⊥︎')
                return alternatives

            r_nxt = '; '.join('{} ∈ [{}, {}]'.format(i, l, u) for i, (l, u) in nxt if l != u)
            print(Fore.YELLOW + '\n||{}||'.format('=' * (len(r_nxt) + 2)))
            print('|| {} ||'.format(r_nxt))
            print('||{}||\n'.format('=' * (len(r_nxt) + 2)), Style.RESET_ALL)
            entry_full = self.initial.assume(nxt)
            r_cstr = ''
            for i, (e, a) in enumerate(constraints):
                r_expr = ' + '.join('({})*{}'.format(v, n) for n, v in e.items() if n != '_')
                r_expr = r_expr + ' + {}'.format(e['_'])
                if a > 0:
                    r_cstr = r_cstr + '[{}+]: '.format(i) + r_expr + ' >= 0\n'
                else:
                    r_cstr = r_cstr + '[{}-]: '.format(i) + r_expr + ' <= 0\n'
            print('Constraints: ', r_cstr)
            status, pattern, picked = self.fwd(entry_full, forced_active=f_active, forced_inactive=f_inactive)

            if status == 1:
                print('✔︎')
                return alternatives
            elif status == -1:
                print('✘')
                return alternatives
            elif status is None:
                print('TODO')
                return alternatives
            else:
                (chosen, expression) = picked
                r_expr = ' + '.join('({})*{}'.format(v, n) for n, v in expression.items() if n != '_')
                r_cstr = r_expr + ' + {}'.format(expression['_'])
                activity = self.original_status(chosen)
                # if constraints and self.is_redundant((expression, -activity), dict([(f[0].name, f[1]) for f in nxt]), constraints):
                #     print('Redundant Constraint: ', r_cstr, ' <= 0')
                #     # the alternative should be unfeasible
                #     # ...
                #     # (_chosen, _constraints, _nxt, _f_active, _f_inactive) = alternatives[-1]
                #     # for name in chosen:
                #     #     _f_inactive.add(self.relus_name2node[VariableIdentifier(name)])
                #     # alternatives[-1] = (_chosen, _constraints, _nxt, _f_active, _f_inactive)
                #     # return alternatives
                # elif constraints and self.is_redundant((expression, activity), dict([(f[0].name, f[1]) for f in nxt]), constraints):
                #     print('Redundant Constraint: ', r_cstr, ' >= 0')
                #     # the alternative should be unfeasible
                #     # ...
                #     # (_chosen, _constraints, _nxt, _f_active, _f_inactive) = alternatives[-1]
                #     # for name in chosen:
                #     #     _f_active.add(self.relus_name2node[VariableIdentifier(name)])
                #     # alternatives[-1] = (_chosen, _constraints, _nxt, _f_active, _f_inactive)
                #     # return alternatives
                alt_f_active, alt_f_inactive = set(f_active), set(f_inactive)
                if activity > 0:  # originally active
                    for name in chosen:
                        f_inactive.add(self.relus_name2node[VariableIdentifier(name)])
                    r_cstr = r_cstr + ' <= 0'
                    for name in chosen:
                        alt_f_active.add(self.relus_name2node[VariableIdentifier(name)])
                else:  # originally inactive
                    assert activity < 0
                    for name in chosen:
                        f_active.add(self.relus_name2node[VariableIdentifier(name)])
                    r_cstr = r_cstr + ' <= 0'
                    for name in chosen:
                        alt_f_inactive.add(self.relus_name2node[VariableIdentifier(name)])
                print('Added Constraint: ', r_cstr)
                alt_constraints = list(constraints)
                alt_constraints.append((expression, activity))
                alt_nxt = self.to_pulp(dict([(f[0].name, f[1]) for f in nxt]), alt_constraints)
                constraints.append((expression, -activity))
                nxt = self.to_pulp(dict([(f[0].name, f[1]) for f in nxt]), constraints)
                alternatives.append((chosen, alt_constraints, alt_nxt, alt_f_active, alt_f_inactive))

    def analyze(self, initial: State):
        print(Fore.BLUE + '\n||==================================||')
        print('|| domain: {}'.format(self.domain))
        print('||==================================||', Style.RESET_ALL)
        self._initial = initial

        print(Fore.MAGENTA + '\n||==========||')
        print('|| Original ||')
        print('||==========||\n', Style.RESET_ALL)
        unperturbed = [(feature, (0, 0)) for feature in self.inputs]
        entry_orig = self.initial.assume(unperturbed)
        status, pattern, _ = self.fwd(entry_orig)
        # set original activation pattern
        self._pattern = list()
        for (a, d, _) in pattern:
            _a = set(self.relus_node2name[n].name for n in a)
            _d = set(self.relus_node2name[n].name for n in d)
            self._pattern.append((_a, _d))

        prefix = list()
        print(Fore.MAGENTA + '\n||========||')
        print('|| Path 0 ||')
        print('||========||\n', Style.RESET_ALL)
        nxt = [(feature, (0, 1)) for feature in self.inputs]
        f_active, f_inactive, constraints, alternatives = set(), set(), list(), list()
        suffix = self.search(nxt, f_active, f_inactive, constraints)
        alternatives = prefix + suffix

        path = 1
        while len(alternatives) > 0 and path < 50:
            print(Fore.MAGENTA + '\n||========||')
            print('|| Path {} ||'.format(path))
            print('||========||\n', Style.RESET_ALL)
            relu, constraints, nxt, f_active, f_inactive = alternatives[-1]
            prefix = alternatives[:-1]
            suffix = self.search(nxt, f_active, f_inactive, constraints)
            alternatives = prefix + suffix
            path = path + 1

        print('Done')

class ActivationPatternForwardSemantics(DefaultForwardSemantics):

    def assume_call_semantics(self, stmt: Call, state: State) -> State:
        argument = self.semantics(stmt.arguments[0], state).result
        state.assume(argument)
        state.result = set()
        return state