"""Tool for reading and returning the contents of a file."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TOOL_SCHEMA = {
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


def cat(path):
    """
    Read and return the full text contents of a file.

    >>> >>> cat('tools/__init__.py')
    '\"\"\"Pack'
    >>> cat('nonexistent_file_xyz.txt')
    'Error: file not found'
    >>> cat('/etc/passwd')
    'Error: unsafe path'
    >>> cat('../secret.txt')
    'Error: unsafe path'
    """
    from chat import is_path_safe
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