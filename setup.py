"""
Setup configuration for OpenGuidance - Advanced AI Guidance System.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'src', 'openguidance', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('"')[1]
    return '1.0.0'

# Read README.md for long description
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements from requirements.txt
def get_requirements():
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read development requirements
def get_dev_requirements():
    dev_requirements_file = os.path.join(os.path.dirname(__file__), 'requirements-dev.txt')
    if os.path.exists(dev_requirements_file):
        with open(dev_requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="openguidance",
    version=get_version(),
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="Advanced AI assistant framework with memory, code execution, and intelligent prompting capabilities",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearchai/OpenGuidance",
    project_urls={
        "Bug Tracker": "https://github.com/llamasearchai/OpenGuidance/issues",
        "Documentation": "https://github.com/llamasearchai/OpenGuidance",
        "Source Code": "https://github.com/llamasearchai/OpenGuidance",
        "Homepage": "https://github.com/llamasearchai/OpenGuidance",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": get_dev_requirements(),
        "full": [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "nltk>=3.6.0",
            "transformers>=4.21.0",
            "redis>=4.0.0",
            "psycopg2-binary>=2.9.0",
            "sqlalchemy>=1.4.0",
            "alembic>=1.8.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn[standard]>=0.20.0",
            "pydantic>=1.10.0",
            "python-multipart>=0.0.6",
        ],
        "monitoring": [
            "prometheus-client>=0.15.0",
            "grafana-api>=1.0.3",
        ],
        "testing": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "httpx>=0.24.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "openguidance=openguidance.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "openguidance": [
            "templates/*.yaml",
            "templates/*.json",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "ai", "artificial intelligence", "gpt", "language model",
        "chatbot", "assistant", "automation", "nlp", "machine learning",
        "prompt engineering", "code generation", "memory", "validation"
    ],
)