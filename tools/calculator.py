import ast, operator

# Safe eval mapping
ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

def calculate(expr: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    node = ast.parse(expr, mode='eval')
    def _eval(n):
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            return ops[type(n.op)](left, right)
        elif isinstance(n, ast.Constant):
            return n.value
        else:
            raise ValueError("Unsupported expression")
    return str(_eval(node.body))