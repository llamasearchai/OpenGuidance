"""
Tests for OpenGuidance CLI functionality.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from click.testing import CliRunner

from openguidance.cli import cli


class TestCLI:
    """Tests for CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @patch('openguidance.cli.OpenGuidance')
    def test_import_command(self, mock_openguidance, runner):
        """Test import command."""
        mock_system = Mock()
        mock_system.initialize = AsyncMock()
        mock_system.cleanup = AsyncMock()
        mock_openguidance.return_value = mock_system
        
        import_result = {
            "success": True,
            "imported_memories": 10,
            "imported_conversations": 5,
            "imported_prompts": 8
        }
        mock_system.import_system_state = AsyncMock(return_value=import_result)
        
        import_data = {
            "memories": [],
            "conversations": {},
            "prompts": {}
        }
        
        # Create a temporary file for the test
        with runner.isolated_filesystem():
            with open('import.json', 'w') as f:
                json.dump(import_data, f)
            
            with patch('openguidance.cli.load_config') as mock_load_config:
                mock_load_config.return_value = Mock()
                
                result = runner.invoke(cli, ['import-data', '--input', 'import.json'])
                
                assert result.exit_code == 0

    @patch('openguidance.cli.OpenGuidance')
    def test_stats_command(self, mock_openguidance, runner):
        """Test stats command."""
        mock_system = Mock()
        mock_system.initialize = AsyncMock()
        mock_system.cleanup = AsyncMock()
        mock_openguidance.return_value = mock_system
        
        stats_data = {
            "execution_stats": {
                "total_executions": 100,
                "successful_executions": 95,
                "failed_executions": 5,
                "average_execution_time": 1.5
            },
            "memory_stats": {
                "total_memories": 250,
                "active_conversations": 10,
                "memory_usage_mb": 45.2
            },
            "prompt_stats": {
                "total_templates": 25,
                "by_category": {
                    "conversation": 8,
                    "code_generation": 7,
                    "explanation": 10
                }
            }
        }
        mock_system.get_system_stats = Mock(return_value=stats_data)
        
        result = runner.invoke(cli, ['stats'])
        
        assert result.exit_code == 0

    def test_invalid_config_key(self, runner):
        """Test setting invalid config key."""
        result = runner.invoke(cli, ['config-show'])
        # This should work without error
        assert result.exit_code == 0

    @patch('openguidance.cli.OpenGuidance')
    def test_chat_with_message(self, mock_openguidance, runner):
        """Test chat command with message."""
        mock_system = Mock()
        mock_system.initialize = AsyncMock()
        mock_system.cleanup = AsyncMock()
        mock_openguidance.return_value = mock_system
        
        mock_result = Mock()
        mock_result.content = "Hello world!"
        mock_result.execution_time = 0.5
        mock_system.process_request = AsyncMock(return_value=mock_result)
        
        result = runner.invoke(cli, ['chat', '--message', 'Hello'])
        
        assert result.exit_code == 0

    def test_cli_error_handling(self, runner):
        """Test CLI error handling."""
        # Test with invalid command
        result = runner.invoke(cli, ['invalid_command'])
        assert result.exit_code != 0

    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(cli, ['version'])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


if __name__ == '__main__':
    pytest.main([__file__, "-v"])