"""Tool for listing files in a directory, similar to the shell ls command."""

import glob
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TOOL_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'ls',
        'description': 'List files and folders in a directory. Optionally takes a path argument.',
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The directory path to list. Defaults to the current directory.',
                },
            },
            'required': [],
        },
    },
}


def ls(path='.'):
    """
    List files and folders in a directory, sorted asciibetically, one per line.

    >>> 'chat.py' in ls('.')
    True
    >>> 'ls.py' in ls('tools')
    True
    >>> ls('/etc')
    'Error: unsafe path'
    >>> ls('../other')
    'Error: unsafe path'
    >>> ls('nonexistent_folder_xyz')
    ''
    """
    from chat import is_path_safe
    if not is_path_safe(path):
        return 'Error: unsafe path'
    files = sorted(glob.glob(f'{path}/*'))
    names = [os.path.basename(f) for f in files]
    return '\n'.join(names)