import ast
import json
import os
import time
from enum import Enum
from typing import Dict

from apronpy.environment import PyEnvironment
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from pip._vendor.colorama import Style

from agbs.abstract_domains.interval_domain import Box2State
from agbs.abstract_domains.symbolic_domain import SymbolicState
from agbs.core.cfg import Activation, Node, Function
from agbs.core.statements import Assignment, Lyra2APRON
from agbs.engine.agbs_interpreter import AGBSInterpreter, ActivationPatternForwardSemantics
from queue import Queue

from agbs.frontend.cfg_generator import ast_to_cfg, VariableIdentifier


class AbstractDomain(Enum):
    BOXES2 = 1
    SYMBOLIC3 = 2


class AGBSRunner:

    def __init__(self, domain=AbstractDomain.SYMBOLIC3):
        self._path = None
        self._cfg = None
        self._domain = domain
        self._inputs = None
        self._relus: Dict[VariableIdentifier, Node] = None
        self._outputs = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        self._cfg = cfg

    @property
    def domain(self):
        return self._domain

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def relus(self):
        return self._relus

    @relus.setter
    def relus(self, relus):
        self._relus = relus

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    def interpreter(self):
        nodes = (self.inputs, self.relus, self.outputs)
        return AGBSInterpreter(self.cfg, self.domain, ActivationPatternForwardSemantics(), nodes)

    def state(self):
        if self.domain == AbstractDomain.SYMBOLIC3:
            state = SymbolicState(self.inputs)
        else:
            state = Box2State(self.inputs)
        return state

    @property
    def variables(self):
        variables, assigned, outputs = set(), set(), set()
        relus: Dict[VariableIdentifier, Node] = dict()
        worklist = Queue()
        worklist.put(self.cfg.in_node)
        while not worklist.empty():
            current = worklist.get()
            if isinstance(current, Function):
                outputs = set()
            for stmt in current.stmts:
                variables = variables.union(stmt.ids())
                if isinstance(stmt, Assignment):
                    assigned = assigned.union(stmt.left.ids())
                    outputs = outputs.union(stmt.left.ids())
                elif isinstance(current, Activation):
                    relus[current.stmts[0].arguments[0].variable] = current
            for node in self.cfg.successors(current):
                worklist.put(node)
        return variables.difference(assigned), variables, outputs, relus

    _lyra2apron = Lyra2APRON()

    def lyra2apron(self, environment):
        layer: int = 1
        worklist = Queue()
        worklist.put(self.cfg.in_node)
        while not worklist.empty():
            current: Node = worklist.get()  # retrieve the current node
            # execute block
            if isinstance(current, Function):
                affine: Dict[VariableIdentifier, PyTexpr1] = dict()
                vars = list()
                exprs = list()
                for assignment in current.stmts:
                    variable, expression = self._lyra2apron.visit(assignment, environment)
                    affine[assignment.left.variable] = expression
                    vars.append(variable)
                    exprs.append(expression)
                layer += 1
                newnode = Function(current.identifier, (vars, exprs))
                self.cfg.nodes[current.identifier] = newnode
            elif isinstance(current, Activation):
                variable = self._lyra2apron.visit(current.stmts[0], environment)
                newnode = Activation(current.identifier, variable)
                self.cfg.nodes[current.identifier] = newnode
            # update worklist
            for node in self.cfg.successors(current):
                worklist.put(node)

    def main(self, path):
        self.path = path
        with open(self.path, 'r') as source:
            self.cfg = ast_to_cfg(ast.parse(source.read()))

            self.inputs, variables, self.outputs, self.relus = self.variables
            r_vars = list()
            for variable in variables:
                r_vars.append(PyVar(variable.name))
            environment = PyEnvironment([], r_vars)
            self.lyra2apron(environment)
        self.run()

    def run(self):
        start = time.time()
        interpreter = self.interpreter()
        state = self.state()
        interpreter.analyze(state)
        end = time.time()
        print('Time: {}s'.format(end - start), Style.RESET_ALL)
