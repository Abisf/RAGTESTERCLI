"""
Unit tests for config module.

Tests YAML config loading, parsing, and error handling.
"""

import pytest
import os
import tempfile
import yaml
from unittest.mock import patch, mock_open
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, config


class TestConfigLoading:
    """Test config loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config_valid_yaml(self):
        """Test loading valid YAML configuration."""
        test_config = {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'api_key': 'test-key',
                'temperature': 0.0
            },
            'output': {
                'format': 'json',
                'include_reasoning': True
            }
        }
        
        yaml_content = yaml.dump(test_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                llm_config = config_obj.get_llm_config()
                
                assert llm_config['provider'] == 'openai'
                assert llm_config['model'] == 'gpt-3.5-turbo'
                assert llm_config['api_key'] == 'test-key'
    
    def test_load_config_missing_file(self):
        """Test loading config when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            config_obj = Config('nonexistent.yaml')
            llm_config = config_obj.get_llm_config()
            
            # Should return default config
            assert isinstance(llm_config, dict)
            assert 'provider' in llm_config
            assert llm_config['provider'] == 'openai'
    
    def test_load_config_invalid_yaml(self):
        """Test loading config with invalid YAML."""
        invalid_yaml = """
        llm:
            provider: openai
            model: gpt-3.5-turbo
            api_key: test-key
        output:
            format: [json, incomplete
        """  # Missing closing bracket
        
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                llm_config = config_obj.get_llm_config()
                
                # Should return default config on YAML error
                assert isinstance(llm_config, dict)
                assert 'provider' in llm_config
    
    def test_load_config_empty_file(self):
        """Test loading config from empty file."""
        with patch('builtins.open', mock_open(read_data='')):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                llm_config = config_obj.get_llm_config()
                
                # Should return default config
                assert isinstance(llm_config, dict)
                assert 'provider' in llm_config
    
    def test_config_with_environment_variables(self):
        """Test config loading with environment variable substitution."""
        test_config = {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'temperature': 0.1
            }
        }
        
        yaml_content = yaml.dump(test_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pathlib.Path.exists', return_value=True):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'}):
                    config_obj = Config('test_config.yaml')
                    llm_config = config_obj.get_llm_config()
                    
                    assert llm_config['api_key'] == 'sk-test-key'
    
    def test_config_find_file(self):
        """Test config file finding functionality."""
        with patch('pathlib.Path.exists') as mock_exists:
            # First file doesn't exist, second does
            mock_exists.side_effect = [False, True, False, False]
            
            config_obj = Config()
            assert config_obj.config_path == 'ragtester.yml'


class TestConfigAccess:
    """Test config access functionality."""
    
    def test_get_llm_config_defaults(self):
        """Test getting default LLM config."""
        with patch('pathlib.Path.exists', return_value=False):
            config_obj = Config()
            llm_config = config_obj.get_llm_config()
            
            assert llm_config['provider'] == 'openai'
            assert llm_config['model'] == 'gpt-4'
            assert llm_config['temperature'] == 0.0
            assert llm_config['max_tokens'] == 1000
    
    def test_get_llm_config_with_overrides(self):
        """Test getting LLM config with file overrides."""
        test_config = {
            'llm': {
                'provider': 'anthropic',
                'model': 'claude-3-haiku',
                'temperature': 0.5
            }
        }
        
        yaml_content = yaml.dump(test_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                llm_config = config_obj.get_llm_config()
                
                assert llm_config['provider'] == 'anthropic'
                assert llm_config['model'] == 'claude-3-haiku'
                assert llm_config['temperature'] == 0.5
                assert llm_config['max_tokens'] == 1000  # Default value
    
    def test_get_output_config_defaults(self):
        """Test getting default output config."""
        with patch('pathlib.Path.exists', return_value=False):
            config_obj = Config()
            output_config = config_obj.get_output_config()
            
            assert output_config['format'] == 'json'
            assert output_config['include_reasoning'] == False
            assert output_config['decimal_places'] == 3
    
    def test_get_output_config_with_overrides(self):
        """Test getting output config with file overrides."""
        test_config = {
            'output': {
                'format': 'csv',
                'include_reasoning': True,
                'decimal_places': 5
            }
        }
        
        yaml_content = yaml.dump(test_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                output_config = config_obj.get_output_config()
                
                assert output_config['format'] == 'csv'
                assert output_config['include_reasoning'] == True
                assert output_config['decimal_places'] == 5


class TestConfigDefaults:
    """Test config default values."""
    
    def test_global_config_instance(self):
        """Test that global config instance works."""
        # Test the actual config object
        assert isinstance(config.get_llm_config(), dict)
        assert isinstance(config.get_output_config(), dict)
        
        llm_config = config.get_llm_config()
        assert 'provider' in llm_config
        assert 'model' in llm_config
        assert 'api_key' in llm_config
        assert 'temperature' in llm_config
        assert 'max_tokens' in llm_config
    
    def test_default_llm_config(self):
        """Test default LLM configuration."""
        llm_config = config.get_llm_config()
        assert llm_config['provider'] == 'openai'
        assert llm_config['model'] == 'gpt-4'
        assert llm_config['temperature'] == 0.0
        assert llm_config['max_tokens'] == 1000
    
    def test_default_output_config(self):
        """Test default output configuration."""
        output_config = config.get_output_config()
        assert output_config['format'] == 'json'
        assert output_config['include_reasoning'] == False
        assert output_config['decimal_places'] == 3


class TestConfigEnvironmentIntegration:
    """Test config integration with environment variables."""
    
    def test_config_environment_variable_loading(self):
        """Test that config loads environment variables."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'}):
            config_obj = Config()
            llm_config = config_obj.get_llm_config()
            
            assert llm_config['api_key'] == 'sk-test-key'
    
    def test_config_missing_environment_variable(self):
        """Test config behavior when environment variable is missing."""
        with patch.dict(os.environ, {}, clear=True):
            config_obj = Config()
            llm_config = config_obj.get_llm_config()
            
            # Should be None when no API key in environment
            assert llm_config['api_key'] is None


class TestConfigErrorHandling:
    """Test config error handling scenarios."""
    
    def test_config_file_permission_error(self):
        """Test handling of file permission errors."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                llm_config = config_obj.get_llm_config()
                
                # Should return default config on permission error
                assert isinstance(llm_config, dict)
                assert 'provider' in llm_config
    
    def test_config_file_unicode_error(self):
        """Test handling of Unicode errors in config file."""
        with patch('builtins.open', side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid utf-8")):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                llm_config = config_obj.get_llm_config()
                
                # Should return default config on Unicode error
                assert isinstance(llm_config, dict)
                assert 'provider' in llm_config
    
    def test_config_yaml_parser_error(self):
        """Test handling of YAML parser errors."""
        invalid_yaml = """
        llm:
            provider: openai
            model: gpt-3.5-turbo
            api_key: "unclosed string
        """
        
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                llm_config = config_obj.get_llm_config()
                
                # Should return default config on YAML parser error
                assert isinstance(llm_config, dict)
                assert 'provider' in llm_config
    
    def test_config_complex_yaml_structure(self):
        """Test handling of complex YAML structures."""
        complex_config = {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'temperature': 0.1,
                'max_tokens': 1000,
                'settings': {
                    'timeout': 30,
                    'retry_attempts': 3
                }
            },
            'output': {
                'format': 'json',
                'include_reasoning': True,
                'decimal_places': 4
            }
        }
        
        yaml_content = yaml.dump(complex_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pathlib.Path.exists', return_value=True):
                config_obj = Config('test_config.yaml')
                
                llm_config = config_obj.get_llm_config()
                output_config = config_obj.get_output_config()
                
                assert llm_config['temperature'] == 0.1
                assert llm_config['settings']['timeout'] == 30
                assert output_config['decimal_places'] == 4


class TestConfigFileDiscovery:
    """Test config file discovery functionality."""
    
    def test_find_config_file_first_found(self):
        """Test finding the first available config file."""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = [True, False, False, False]
            
            config_obj = Config()
            assert config_obj.config_path == 'ragtester.yaml'
    
    def test_find_config_file_home_directory(self):
        """Test finding config file in home directory."""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = [False, False, True, False]
            
            config_obj = Config()
            assert config_obj.config_path == os.path.expanduser("~/.ragtester.yaml")
    
    def test_find_config_file_none_found(self):
        """Test behavior when no config file is found."""
        with patch('pathlib.Path.exists', return_value=False):
            config_obj = Config()
            assert config_obj.config_path is None
            
            # Should still work with defaults
            llm_config = config_obj.get_llm_config()
            assert llm_config['provider'] == 'openai'