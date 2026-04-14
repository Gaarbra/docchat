"""Package setup for docchat."""

from setuptools import setup, find_packages

setup(
    name='docchat',
    version='1.0.0',
    description='An AI-powered terminal agent for chatting with documents.',
    packages=find_packages(),
    install_requires=[
        'groq>=0.9.0',
    ],
    entry_points={
        'console_scripts': [
            'chat=chat:main',
        ],
    },
    python_requires='>=3.8',
)