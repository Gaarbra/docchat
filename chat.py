"""AI-powered document chat agent.

Lets users query files using natural language and shell-like commands.
"""

import ast
import glob
import json
import operator
import os
import re
import sys

from groq import Groq

MODEL = 'llama-3.3-70b-versatile'

CALCULATE_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'calculate',
        'description': (
            'Evaluate a simple arithmetic expression and '
            'return the result.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'expression': {
                    'type': 'string',
                    'description': (
                        'The arithmetic expression to evaluate, '
                        'e.g. "2 + 2" or "10 * (3 + 4)".'
                    ),
                },
            },
            'required': ['expression'],
        },
    },
}

LS_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'ls',
        'description': (
            'List files and folders in a directory. '
            'Optionally takes a path argument.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': (
                        'The directory path to list. Defaults to '
                        'the current directory.'
                    ),
                },
            },
            'required': [],
        },
    },
}

CAT_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'cat',
        'description': 'Read and return the contents of a text file.',
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The path to the file to read.',
                },
            },
            'required': ['path'],
        },
    },
}

GREP_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'grep',
        'description': (
            'Search for lines matching a regex pattern '
            'in a file or directory.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'The regex pattern to search for.',
                },
                'path': {
                    'type': 'string',
                    'description': (
                        'The file or directory path to search. '
                        'Defaults to current directory.'
                    ),
                },
            },
            'required': ['pattern'],
        },
    },
}

ALL_TOOL_SCHEMAS = [CALCULATE_SCHEMA, LS_SCHEMA, CAT_SCHEMA, GREP_SCHEMA]

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
    Recursively evaluate a single AST node.

    Raises ValueError for unsafe expressions.

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


def is_path_safe(path):
    """
    Returns True if a path is safe to read.

    Checks for absolute paths or directory traversal.

    >>> is_path_safe('README.md')
    True
    >>> is_path_safe('chat.py')
    True
    >>> is_path_safe('/etc/passwd')
    False
    >>> is_path_safe('../secret.txt')
    False
    >>> is_path_safe('some/../file.txt')
    False
    >>> is_path_safe('.')
    True
    >>> is_path_safe('')
    True
    """
    if path.startswith('/'):
        return False
    parts = path.replace('\\', '/').split('/')
    if '..' in parts:
        return False
    return True


class Chat:
    """
    A chat agent answering questions about files via tool calls.

    Supports four tools: calculate, ls, cat, and grep. Users can invoke
    tools manually with /command syntax or let the LLM call automatically.

    >>> c = Chat()
    >>> isinstance(c.messages, list)
    True
    >>> len(c.messages)
    0
    >>> result = c.run_tool('ls', {'path': '.'})
    >>> 'chat.py' in result
    True
    >>> result = c.run_tool('cat', {'path': 'chat.py'})
    >>> isinstance(result, str)
    True
    >>> result = c.run_tool('calculate', {'expression': '2 + 2'})
    >>> '4' in result
    True
    >>> result = c.run_tool('ls', {})
    >>> 'chat.py' in result
    True
    """

    def __init__(self):
        """Initialize Chat with empty message history and Groq client."""
        self.messages = []
        self.client = Groq()
        self.tool_dispatch = {
            'calculate': self.calculate,
            'ls': self.ls,
            'cat': self.cat,
            'grep': self.grep,
        }

    def calculate(self, expression):
        """
        Evaluate a simple arithmetic expression and return the result.

        Supports +, -, *, /, //, %, **, and parentheses.
        Rejects anything unsafe.

        >>> c = Chat()
        >>> c.calculate('2 + 2')
        '4'
        >>> c.calculate('10 * 5')
        '50'
        >>> c.calculate('100 / 4')
        '25.0'
        >>> c.calculate('2 ** 8')
        '256'
        >>> c.calculate('(3 + 4) * 2')
        '14'
        >>> c.calculate('10 % 3')
        '1'
        >>> c.calculate('10 // 3')
        '3'
        >>> c.calculate('-5 + 3')
        '-2'
        >>> c.calculate('1 / 0')
        'Error: division by zero'
        >>> c.calculate('__import__("os")')
        'Error: invalid expression'
        >>> c.calculate('abc')
        'Error: invalid expression'
        >>> c.calculate('open("file")')
        'Error: invalid expression'
        """
        try:
            tree = ast.parse(expression, mode='eval')
            result = _eval_node(tree.body)
            if isinstance(result, float) and result.is_integer():
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

    def cat(self, path):
        """
        Read and return the full text contents of a file.

        >>> c = Chat()
        >>> c.cat('chat.py')[:7]
        '\"\"\"AI-p'
        >>> c.cat('nonexistent_file_xyz.txt')
        'Error: file not found'
        >>> c.cat('/etc/passwd')
        'Error: unsafe path'
        >>> c.cat('../secret.txt')
        'Error: unsafe path'
        """
        if not is_path_safe(path):
            return 'Error: unsafe path'
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return 'Error: file not found'
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='utf-16') as f:
                    return f.read()
            except Exception:
                return 'Error: cannot decode file'
        except Exception as e:
            return f'Error: {e}'

    def grep(self, pattern, path='.'):
        """
        Search for lines matching a regex pattern (recursive).

        Returns matching lines as 'filename:line', or an error string.

        >>> c = Chat()
        >>> result = c.grep('def is_path_safe', 'chat.py')
        >>> 'chat.py' in result
        True
        >>> c.grep('def ', '/etc')
        'Error: unsafe path'
        >>> c.grep('def ', '../other')
        'Error: unsafe path'
        >>> c.grep('zzz[n]omatch_xyz', 'chat.py')
        ''
        >>> c.grep('[invalid', 'chat.py')
        'Error: invalid pattern: unterminated character set at position 0'
        """
        if not is_path_safe(path):
            return 'Error: unsafe path'
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f'Error: invalid pattern: {e}'

        results = []
        if os.path.isfile(path):
            files = [path]
        else:
            files = []
            for root, dirs, filenames in os.walk(path):
                dirs[:] = sorted([d for d in dirs if not d.startswith('.')])
                for fname in sorted(filenames):
                    files.append(os.path.join(root, fname))

        for filepath in files:
            try:
                with open(
                    filepath, 'r', encoding='utf-8', errors='ignore'
                ) as f:
                    for line in f:
                        if compiled.search(line):
                            results.append(f'{filepath}:{line.rstrip()}')
            except Exception:
                continue

        return '\n'.join(results)

    def ls(self, path='.'):
        """
        List files/folders in a directory, asciibetically, one per line.

        >>> c = Chat()
        >>> 'chat.py' in c.ls('.')
        True
        >>> c.ls('/etc')
        'Error: unsafe path'
        >>> c.ls('../other')
        'Error: unsafe path'
        >>> c.ls('nonexistent_folder_xyz')
        ''
        """
        if not is_path_safe(path):
            return 'Error: unsafe path'
        files = sorted(glob.glob(f'{path}/*'))
        names = [os.path.basename(f) for f in files]
        return '\n'.join(names)

    def run_tool(self, name, args):
        """
        Dispatch a tool call by name and return its string output.

        >>> c = Chat()
        >>> result = c.run_tool('ls', {'path': '.'})
        >>> 'chat.py' in result
        True
        >>> c.run_tool('unknown_tool', {})
        'Error: unknown tool unknown_tool'
        >>> result = c.run_tool('calculate', {'expression': '10 * 5'})
        >>> '50' in result
        True
        """
        if name not in self.tool_dispatch:
            return f'Error: unknown tool {name}'
        func = self.tool_dispatch[name]
        try:
            return func(**args)
        except Exception as e:
            return f'Error: {e}'

    def chat(self, user_message):
        """
        Send a user message to the LLM and handle any tool call loops.

        Returns the final response.

        >>> c = Chat()
        >>> response = c.chat('Say only the word HELLO and nothing else.')
        >>> 'HELLO' in response
        True
        >>> len(c.messages)
        2
        """
        self.messages.append({'role': 'user', 'content': user_message})
        while True:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=ALL_TOOL_SCHEMAS,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self.messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = self.run_tool(tc.function.name, args)
                    self.messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.id,
                        'content': result,
                    })
            else:
                content = msg.content or ''
                self.messages.append({'role': 'assistant', 'content': content})
                return content

    def repl(self):
        """
        Run the interactive read-eval-print loop for the chat agent.

        This function is not tested via doctests because it performs IO.
        """
        print('docchat - type /help for commands, Ctrl+C to exit')
        while True:
            try:
                user_input = input('chat> ').strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if not user_input:
                continue

            if user_input.startswith('/'):
                parts = user_input[1:].split()
                if not parts:
                    continue
                tool_name = parts[0]
                if tool_name == 'help':
                    print('Available commands: /calculate, /ls, /cat, /grep')
                    continue
                args_list = parts[1:]
                
                if tool_name == 'ls':
                    kwargs = {'path': args_list[0]} if args_list else {}
                elif tool_name == 'cat':
                    kwargs = {'path': args_list[0]} if args_list else {}
                elif tool_name == 'grep':
                    if len(args_list) >= 2:
                        kwargs = {
                            'pattern': args_list[0],
                            'path': args_list[1]
                        }
                    elif len(args_list) == 1:
                        kwargs = {'pattern': args_list[0], 'path': '.'}
                    else:
                        print('Usage: /grep <pattern> <path>')
                        continue
                elif tool_name == 'calculate':
                    kwargs = {'expression': ' '.join(args_list)}
                else:
                    print(f'Unknown command: {tool_name}')
                    continue

                result = self.run_tool(tool_name, kwargs)
                print(result)
                
                self.messages.append({
                    'role': 'user',
                    'content': (
                        f'[manual command] /{tool_name} '
                        f'{" ".join(args_list)}\nOutput:\n{result}'
                    ),
                })
                self.messages.append({
                    'role': 'assistant',
                    'content': (
                        f'I ran `/{tool_name} {" ".join(args_list)}` '
                        f'and got:\n{result}'
                    ),
                })
            else:
                response = self.chat(user_input)
                print(response)


def main():
    """
    Entry point: accept CLI args or start the repl.

    This function is not tested via doctests because it performs IO.
    """
    c = Chat()
    if len(sys.argv) > 1:
        message = ' '.join(sys.argv[1:])
        response = c.chat(message)
        print(response)
    else:
        c.repl()


if __name__ == '__main__':
    main()