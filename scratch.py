from csp.grid import parse
from csp.solver import Solver

p = parse("530070000600195000098000060800060003400803001"
          "700020006060000280000419005000080079")
for fc in (False, True):
    s = Solver(use_fc=fc)
    s.solve(p)
    print("FC" if fc else "BT", s.stats)