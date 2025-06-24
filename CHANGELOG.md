# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Advanced memory system with multiple backends
- Code execution sandbox with security features
- Intelligent prompt template system
- Streaming response support
- Batch processing capabilities
- Docker and Docker Compose support
- Comprehensive CLI interface
- Performance monitoring and statistics
- Export/import functionality
- Multi-session conversation management

### Changed
- Improved async architecture for better performance
- Enhanced error handling and recovery
- Better type hints and documentation
- Optimized memory usage and caching

### Security
- Sandboxed code execution environment
- Input validation and sanitization
- Rate limiting protection
- Secure credential management

## [0.1.0] - 2024-01-01

### Added
- Initial release of OpenGuidance
- Core system architecture
- Basic memory management
- Simple prompt system
- Code execution capabilities
- CLI interface
- Docker support
- Comprehensive test suite
- Documentation and examples

### Features
- GPT-4 integration
- Conversation memory
- Python code execution
- Template-based prompts
- Async processing
- Configuration management
- Error handling
- Performance monitoring

### Development
- Pre-commit hooks
- CI/CD pipeline
- Code coverage reporting
- Type checking
- Linting and formatting
- Docker builds
- PyPI publishing
```

```markdown:CONTRIBUTING.md
# Contributing to OpenGuidance

Thank you for your interest in contributing to OpenGuidance! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@openguidance.ai.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
- Check if the issue has already been reported
- Include as much detail as possible
- Provide steps to reproduce the issue

Use the bug report template:

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What you expected to happen.

**Actual Behavior**  
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.11]
- OpenGuidance version: [e.g., 0.1.0]
```

### Suggesting Features

Feature requests are welcome! Please:
- Check if the feature has been suggested before
- Explain the use case clearly
- Consider the impact on existing functionality
- Provide implementation ideas if possible

### Development Setup

1. **Fork the repository**
```bash
git clone https://github.com/your-username/openguidance.git
cd openguidance
```

2. **Set up development environment**
```bash
make dev-setup
```

3. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

4. **Make your changes**
- Follow the coding standards
- Add tests for new functionality
- Update documentation as needed

5. **Run tests and checks**
```bash
make dev-test
```

6. **Commit your changes**
```bash
git add .
git commit -m "feat: add your feature description"
```

7. **Push and create a pull request**
```bash
git push origin feature/your-feature-name
```

### Coding Standards

#### Python Code Style
- Follow PEP 8
- Use Black for formatting
- Maximum line length: 88 characters
- Use type hints for all functions
- Write docstrings for all public methods

#### Example:
```python
def process_request(
    self, 
    request: str, 
    session_id: str,
    timeout: Optional[float] = None
) -> ExecutionResult:
    """
    Process a user request and return the result.
    
    Args:
        request: The user's request text
        session_id: Unique identifier for the session
        timeout: Optional timeout in seconds
        
    Returns:
        ExecutionResult containing the response
        
    Raises:
        ExecutionError: If processing fails
    """
    pass
```

#### Documentation
- Use docstrings for all modules, classes, and functions
- Include type information and examples
- Update README.md for significant changes
- Add inline comments for complex logic

#### Testing
- Write tests for all new functionality
- Maintain test coverage above 90%
- Use descriptive test names
- Include both unit and integration tests

Example test:
```python
@pytest.mark.asyncio
async def test_process_request_success():
    """Test successful request processing."""
    config = Config(model_name="test-model", api_key="test-key")
    guidance = OpenGuidance(config)
    
    with patch.object(guidance.execution_engine, 'execute_request') as mock_execute:
        mock_result = ExecutionResult(
            content="Test response",
            status=ExecutionStatus.COMPLETED,
            execution_time=1.0,
            session_id="test"
        )
        mock_execute.return_value = mock_result
        
        result = await guidance.process_request("test", "session_1")
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.content == "Test response"
```

### Git Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

Examples:
```
feat(memory): add Redis backend support
fix(execution): handle timeout errors properly
docs(api): update configuration examples
```

### Pull Request Process

1. **Ensure all checks pass**
   - Tests pass
   - Code coverage maintained
   - Linting passes
   - Type checking passes

2. **Update documentation**
   - Update docstrings
   - Update README if needed
   - Add changelog entry

3. **Write a clear PR description**
   - Explain what changes were made
   - Reference related issues
   - Include testing details

4. **Request review**
   - Assign reviewers
   - Respond to feedback promptly
   - Make requested changes

### Release Process

Releases are handled by maintainers:

1. Update version numbers
2. Update CHANGELOG.md
3. Create release PR
4. Merge to main
5. Tag release
6. Publish to PyPI
7. Update Docker images

## Development Guidelines

### Architecture Principles

1. **Async First**: Use async/await for all I/O operations
2. **Modular Design**: Keep components loosely coupled
3. **Type Safety**: Use type hints everywhere
4. **Error Handling**: Comprehensive error handling and recovery
5. **Performance**: Consider performance implications
6. **Security**: Security-first approach to all features

### Adding New Features

When adding new features:

1. **Design First**: Create an issue with design proposal
2. **Start Small**: Break large features into smaller PRs
3. **Test Thoroughly**: Include comprehensive tests
4. **Document**: Update documentation and examples
5. **Backward Compatibility**: Maintain API compatibility

### Code Review Guidelines

When reviewing code:

1. **Be Constructive**: Provide helpful feedback
2. **Check Functionality**: Ensure code works as intended
3. **Review Tests**: Verify adequate test coverage
4. **Check Documentation**: Ensure documentation is updated
5. **Consider Performance**: Look for performance implications
6. **Security Review**: Check for security issues

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat and community support
- **Email**: Direct communication with maintainers

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Social media announcements

## Getting Help

If you need help:

1. Check the documentation
2. Search existing issues
3. Ask in GitHub Discussions
4. Join our Discord community
5. Email the maintainers

## License

By contributing to OpenGuidance, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to OpenGuidance! [LAUNCH]
```

```bash:scripts/dev-setup.sh
#!/bin/bash

# Development environment setup script for OpenGuidance

set -e

echo "[LAUNCH] Setting up OpenGuidance development environment..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "[ERROR] Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "[SUCCESS] Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[PACKAGE] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "[TOOL] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[UPGRADE] Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "[INSTALL] Installing development dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🎣 Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "[DIRECTORY] Creating project directories..."
mkdir -p data
mkdir -p logs
mkdir -p config
mkdir -p prompts/templates

# Create default configuration
if [ ! -f "config/default.yaml" ]; then
    echo "[CONFIG] Creating default configuration..."
    cat > config/default.yaml << EOF
# Default OpenGuidance Configuration
model_name: "gpt-4"
api_key: ""
temperature: 0.7
max_tokens: 2000
enable_memory: true
enable_code_execution: true

memory_config:
  max_memory_items: 1000
  cleanup_interval: 3600
  storage_backend: "file"

execution_config:
  max_execution_time: 30
  enable_validation: true
  sandbox_timeout: 10

prompt_config:
  template_directory: "./prompts/templates"
  enable_caching: true
EOF
fi

# Create .env template
if [ ! -f ".env.example" ]; then
    echo "[KEY] Creating environment template..."
    cat > .env.example << EOF
# OpenGuidance Environment Variables
OPENGUIDANCE_API_KEY=your-api-key-here
OPENGUIDANCE_MODEL_NAME=gpt-4
OPENGUIDANCE_ENABLE_MEMORY=true
OPENGUIDANCE_ENABLE_CODE_EXECUTION=true

# Optional: Database URLs
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/openguidance

# Optional: Advanced Settings
OPENGUIDANCE_DEBUG=false
OPENGUIDANCE_LOG_LEVEL=INFO
EOF
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
python -m pytest tests/ -x -v --tb=short

# Check code quality
echo "[SEARCH] Running code quality checks..."
black --check openguidance/ tests/ || echo "[INFO] Run 'make format' to fix formatting"
flake8 openguidance/ tests/ || echo "[INFO] Fix linting issues before committing"

echo ""
echo "[CELEBRATION] Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your API keys"
echo "2. Run 'source venv/bin/activate' to activate the environment"
echo "3. Run 'make test' to run the full test suite"
echo "4. Run 'openguidance --help' to see available commands"
echo ""
echo "Happy coding! [LAUNCH]"
```

```bash:scripts/benchmark.sh
#!/bin/bash

# Benchmark script for OpenGuidance performance testing

set -e

echo "[RUNNING] Running OpenGuidance performance benchmarks..."

# Ensure we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "[ERROR] Please activate your virtual environment first"
    exit 1
fi

# Create benchmark results directory
mkdir -p benchmark_results
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="benchmark_results/$timestamp"
mkdir -p "$results_dir"

echo "[CHART] Results will be saved to: $results_dir"

# Memory usage benchmark
echo "[AI] Testing memory usage..."
python -c "
import asyncio
import psutil
import time
from openguidance import OpenGuidance, Config

async def memory_benchmark():
    process = psutil.Process()
    
    # Baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024
    
    config = Config(
        model_name='test-model',
        api_key='test-key',
        enable_memory=True
    )
    
    system = OpenGuidance(config)
    await system.initialize()
    
    # Memory after initialization
    init_memory = process.memory_info().rss / 1024 / 1024
    
    # Process multiple requests
    for i in range(100):
        await system.memory_manager.store_memory(
            {'fact': f'Test fact {i}'},
            'fact',
            importance='medium'
        )
    
    # Memory after processing
    final_memory = process.memory_info().rss / 1024 / 1024
    
    await system.cleanup()
    
    print(f'Baseline memory: {baseline_memory:.2f} MB')
    print(f'After init: {init_memory:.2f} MB')
    print(f'After processing: {final_memory:.2f} MB')
    print(f'Memory overhead: {final_memory - baseline_memory:.2f} MB')

asyncio.run(memory_benchmark())
" > "$results_dir/memory_usage.txt"

# Response time benchmark
echo "[TIMER] Testing response times..."
python -c "
import asyncio
import time
import statistics
from unittest.mock import patch, AsyncMock
from openguidance import OpenGuidance, Config

async def response_time_benchmark():
    config = Config(
        model_name='test-model',
        api_key='test-key'
    )
    
    system = OpenGuidance(config)
    await system.initialize()
    
    # Mock response generation
    with patch.object(
        system.execution_engine.response_generator,
        'generate_response',
        new_callable=AsyncMock,
        return_value='Mock response'
    ):
        times = []
        
        for i in range(50):
            start = time.time()
            await system.process_request(f'Test request {i}', f'session_{i}')
            end = time.time()
            times.append(end - start)
        
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f'Average response time: {avg_time:.3f}s')
        print(f'Median response time: {median_time:.3f}s')
        print(f'Min response time: {min_time:.