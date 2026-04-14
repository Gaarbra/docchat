"""Tool for evaluating arithmetic expressions safely."""

import ast
import operator

TOOL_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'calculate',
        'description': 'Evaluate a simple arithmetic expression and return the result.',
        'parameters': {
            'type': 'object',
            'properties': {
                'expression': {
                    'type': 'string',
                    'description': 'The arithmetic expression to evaluate, e.g. "2 + 2" or "10 * (3 + 4)".',
                },
            },
            'required': ['expression'],
        },
    },
}

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _eval_node(node):
    """
    Recursively evaluate a single AST node, raising ValueError for unsafe expressions.

    >>> _eval_node(ast.parse('2 + 2', mode='eval').body)
    4
    >>> _eval_node(ast.parse('3 * 7', mode='eval').body)
    21
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError('invalid expression')
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError('invalid expression')
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError('invalid expression')
        operand = _eval_node(node.operand)
        return _ALLOWED_OPS[op_type](operand)
    else:
        raise ValueError('invalid expression')


def calculate(expression):
    """
    Evaluate a simple arithmetic expression and return the result as a string.

    Supports +, -, *, /, //, %, **, and parentheses. Rejects anything unsafe.

    >>> calculate('2 + 2')
    '4'
    >>> calculate('10 * 5')
    '50'
    >>> calculate('100 / 4')
    '25.0'
    >>> calculate('2 ** 8')
    '256'
    >>> calculate('(3 + 4) * 2')
    '14'
    >>> calculate('10 % 3')
    '1'
    >>> calculate('10 // 3')
    '3'
    >>> calculate('-5 + 3')
    '-2'
    >>> calculate('1 / 0')
    'Error: division by zero'
    >>> calculate('__import__("os")')
    'Error: invalid expression'
    >>> calculate('abc')
    'Error: invalid expression'
    >>> calculate('open("file")')
    'Error: invalid expression'
    """
    try:
        tree = ast.parse(expression, mode='eval')
        result = _eval_node(tree.body)
        # Return integer string if result is a whole number
        if isinstance(result, float) and result.is_integer():
            # Only simplify if the expression itself used integer division
            if '/' in expression and '//' not in expression:
                return str(result)
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return 'Error: division by zero'
    except (ValueError, TypeError):
        return 'Error: invalid expression'
    except SyntaxError:
        return 'Error: invalid expression'