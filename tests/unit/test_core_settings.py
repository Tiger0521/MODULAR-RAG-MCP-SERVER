"""核心配置模块测试"""

import pytest
import tempfile
import os
from pathlib import Path
from src.core.settings import Settings, load_settings


class TestSettings:
    """配置模块测试类"""
    
    def test_load_default_settings(self):
        """测试默认配置加载"""
        settings = Settings()
        
        assert settings.llm.provider == "openai"
        assert settings.embedding.provider == "openai"
        assert settings.vector_store.provider == "chroma"
        assert settings.processing.chunk_size == 1000
        
    def test_load_from_yaml(self):
        """测试从YAML文件加载配置"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
llm:
  provider: "test-provider"
  model: "test-model"

embedding:
  provider: "test-embedding"
  model: "test-embedding-model"

vector_store:
  provider: "test-vector-store"
""")
            temp_config_path = f.name
        
        try:
            settings = Settings.load_from_yaml(temp_config_path)
            
            assert settings.llm.provider == "test-provider"
            assert settings.llm.model == "test-model"
            assert settings.embedding.provider == "test-embedding"
            assert settings.vector_store.provider == "test-vector-store"
            
        finally:
            # 清理临时文件
            os.unlink(temp_config_path)
    
    def test_validate_settings_success(self):
        """测试配置验证成功"""
        settings = Settings()
        
        # 应该不会抛出异常
        assert settings.validate_settings() is True
    
    def test_global_settings_singleton(self):
        """测试全局配置单例模式"""
        # 第一次加载
        settings1 = load_settings()
        
        # 第二次加载应该返回同一个实例
        settings2 = load_settings()
        
        assert settings1 is settings2
    
    def test_env_var_resolution(self):
        """测试环境变量解析"""
        # 设置环境变量
        os.environ["TEST_API_KEY"] = "test-key-123"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
llm:
  api_key: "${TEST_API_KEY}"
""")
            temp_config_path = f.name
        
        try:
            settings = Settings.load_from_yaml(temp_config_path)
            assert settings.llm.api_key == "test-key-123"
            
        finally:
            os.unlink(temp_config_path)
            # 清理环境变量
            if "TEST_API_KEY" in os.environ:
                del os.environ["TEST_API_KEY"]


if __name__ == "__main__":
    pytest.main([__file__])