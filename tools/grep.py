"""Tool for searching files with a regex pattern, similar to the shell grep command."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOOL_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'grep',
        'description': 'Search for lines matching a regex pattern in a file or directory.',
        'parameters': {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'The regex pattern to search for.',
                },
                'path': {
                    'type': 'string',
                    'description': 'The file or directory path to search. Defaults to current directory.',
                },
            },
            'required': ['pattern'],
        },
    },
}


def grep(pattern, path='.'):
    """
    Search for lines matching a regex pattern in a file or directory (recursive).

    Returns matching lines as 'filename:line', or an error string.

    >>> result = grep('def is_path_safe', 'chat.py')
    >>> 'chat.py' in result
    True
    >>> grep('def ', '/etc')
    'Error: unsafe path'
    >>> grep('def ', '../other')
    'Error: unsafe path'
    >>> grep('zzznomatch_xyz', 'chat.py')
    ''
    >>> grep('[invalid', 'chat.py')
    'Error: invalid pattern: unterminated character set at position 0'
    """
    from chat import is_path_safe
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
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if compiled.search(line):
                        results.append(f'{filepath}:{line.rstrip()}')
        except Exception:
            continue

    return '\n'.join(results)