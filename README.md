# docchat

An AI-powered terminal agent that lets you ask questions about files and documents in natural language using LLM tool calling.

![doctests](https://github.com/YOUR_USERNAME/docchat/actions/workflows/doctests.yml/badge.svg)
![integration tests](https://github.com/YOUR_USERNAME/docchat/actions/workflows/integration_tests.yml/badge.svg)
![flake8](https://github.com/YOUR_USERNAME/docchat/actions/workflows/flake8.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/docchat.svg)](https://badge.fury.io/py/docchat)

<!-- Replace the animated gif below with your own terminal recording -->
![demo](demo.gif)

## Usage

Install dependencies and run from any project folder:

```bash
pip install -r requirements.txt
python chat.py
```

You can also pass a single message directly:

```bash
python chat.py 'what files are in the .github folder?'
```

### Manual tool commands

Inside the REPL you can call tools directly with `/command` syntax for instant feedback:

```
chat> /ls .github
workflows
chat> /cat README.md
# docchat
...
chat> /grep def chat.py
chat.py:def is_path_safe(path):
chat.py:    def __init__(self):
...
chat> /calculate 2 ** 10
1024
```

## Example: Markdown Compiler project

```bash
$ cd test_projects/markdown_compiler
$ python ../../chat.py
chat> does this project use regular expressions?
Let me check the source files for any imports of the `re` module.
No — after grepping all Python files I found no imports of the `re` library.
```

This example is useful because it shows the agent using the grep tool automatically to answer a question about the codebase, without the user needing to know which files to look at.

## Example: Ebay Scraper project

```bash
$ cd test_projects/ebay_scraper
$ python ../../chat.py
chat> tell me about this project
The README says this project scrapes product listings from eBay to compare prices.
chat> is scraping eBay legal?
Generally yes — scraping publicly accessible pages is legal in most jurisdictions,
though eBay's Terms of Service discourage it. Using their official API is the recommended approach.
```

This example is useful because it demonstrates multi-turn conversation where the agent reads the README and then reasons about a follow-up question using its own knowledge.
