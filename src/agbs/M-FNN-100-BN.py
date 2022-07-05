"""
forward runner to toy example
"""
import sys
sys.setrecursionlimit(10000)
from agbs.engine.agbs_analysis import AGBSRunner, AbstractDomain

nn = 'tests/M-FNN-100-BN.py'
domain = AbstractDomain.SYMBOLIC3
print(f"> Domain chosen: '{domain}'")
b = AGBSRunner(domain=domain)
r = b.main(nn)
