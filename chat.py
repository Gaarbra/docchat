"""AI-powered document chat agent that lets users query files using natural language and shell-like commands."""

import json
import os
import sys

from groq import Groq

from tools.calculate import calculate, TOOL_SCHEMA as CALCULATE_SCHEMA
from tools.ls import ls, TOOL_SCHEMA as LS_SCHEMA
from tools.cat import cat, TOOL_SCHEMA as CAT_SCHEMA
from tools.grep import grep, TOOL_SCHEMA as GREP_SCHEMA

MODEL = 'llama-3.3-70b-versatile'

ALL_TOOL_SCHEMAS = [CALCULATE_SCHEMA, LS_SCHEMA, CAT_SCHEMA, GREP_SCHEMA]
TOOL_DISPATCH = {
    'calculate': calculate,
    'ls': ls,
    'cat': cat,
    'grep': grep,
}


def is_path_safe(path):
    """
    Returns True if a path is safe to read (no absolute paths, no directory traversal).

    >>> is_path_safe('README.md')
    True
    >>> is_path_safe('tools/ls.py')
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
    A chat agent that can answer questions about files using tool calls.

    The agent supports four tools: calculate, ls, cat, and grep.
    Users can invoke tools manually with /command syntax or let the LLM call them automatically.

    >>> c = Chat()
    >>> isinstance(c.messages, list)
    True
    >>> len(c.messages)
    0
    >>> result = c.run_tool('ls', {'path': '.'})
    >>> 'chat.py' in result
    True
    >>> result = c.run_tool('cat', {'path': 'README.md'})
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
        """Initialize a Chat instance with an empty message history and a Groq client."""
        self.messages = []
        self.client = Groq()

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
        if name not in TOOL_DISPATCH:
            return f'Error: unknown tool {name}'
        func = TOOL_DISPATCH[name]
        try:
            return func(**args)
        except Exception as e:
            return f'Error: {e}'

    def chat(self, user_message):
        """
        Send a user message to the LLM and handle any tool call loops, returning the final response.

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
                # Build kwargs based on tool
                if tool_name == 'ls':
                    kwargs = {'path': args_list[0]} if args_list else {}
                elif tool_name == 'cat':
                    kwargs = {'path': args_list[0]} if args_list else {}
                elif tool_name == 'grep':
                    if len(args_list) >= 2:
                        kwargs = {'pattern': args_list[0], 'path': args_list[1]}
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
                # Add to message history so LLM has context
                self.messages.append({
                    'role': 'user',
                    'content': f'[manual command] /{tool_name} {" ".join(args_list)}\nOutput:\n{result}',
                })
                self.messages.append({
                    'role': 'assistant',
                    'content': f'I ran `/{tool_name} {" ".join(args_list)}` and got:\n{result}',
                })
            else:
                response = self.chat(user_input)
                print(response)


def main():
    """
    Entry point: optionally accept a message as a CLI argument, otherwise start the repl.

    This function is not tested via doctests because it performs IO.
    """
    c = Chat()
    if len(sys.argv) > 1:
        # Extra credit: single message mode
        message = ' '.join(sys.argv[1:])
        response = c.chat(message)
        print(response)
    else:
        c.repl()


if __name__ == '__main__':
    main()