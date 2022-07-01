from copy import deepcopy
from queue import Queue
from typing import Set, Dict

from pip._vendor.colorama import Fore, Style

from agbs.abstract_domains.state import State
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

    def fwd(self, initial, forced_active=None, forced_inactive=None):
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
                state = state.affine(current.stmts[0], current.stmts[1])
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

        state.log(self.outputs)

        absolute = dict((k, abs(v)) for k, v in state.polarities.items())  # empty if all relus are fixed
        balanced = min(absolute, key=absolute.get) if absolute else None
        polarity = absolute[balanced] if balanced else None
        symbols = state.expressions[balanced] if balanced else None

        return activations, (balanced, polarity, symbols)

    def print_pattern(self, activations):
        activated, deactivated, uncertain = 0, 0, 0
        print('Activation Pattern: {', end='')
        for (a, d, u) in activations:
            print('[')
            activated += len(a)
            print('activated: ', ','.join(self.relus_node2name[n].name for n in a))
            deactivated += len(d)
            print('deactivated: ', ','.join(self.relus_node2name[n].name for n in d))
            uncertain += len(u)
            print('uncertain: ', ','.join(self.relus_node2name[n].name for n in u))
            print(']', end='')
        print('}')
        print('#Active: ', activated, '#Inactive: ', deactivated, '#Uncertain: ', uncertain, '\n')

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
        pattern, (balanced, polarity, symbols) = self.fwd(entry_orig)
        self.print_pattern(pattern)

        print(Fore.MAGENTA + '\n||======||')
        print('|| Full ||')
        print('||======||\n', Style.RESET_ALL)
        full = [(feature, (0, 1)) for feature in self.inputs]
        entry_full = self.initial.assume(full)
        pattern, (balanced, polarity, symbols) = self.fwd(entry_full)
        self.print_pattern(pattern)

        # print('Total ReLUs: ', len(self.relus))
        # activated, deactivated, uncertain, (balanced, polarity, symbols) = self.fwd(initial)



class ActivationPatternForwardSemantics(DefaultForwardSemantics):

    def assume_call_semantics(self, stmt: Call, state: State) -> State:
        argument = self.semantics(stmt.arguments[0], state).result
        state.assume(argument)
        state.result = set()
        return state