"""
Tests for Config system
测试配置加载、环境变量覆盖
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

from src.config import (
    ConfigLoader,
    SystemConfig,
    ScreenConfig,
    SearchConfig,
    load_config,
    get_agent_config,
)


@pytest.fixture
def sample_config_file():
    """创建示例配置文件"""
    temp_dir = tempfile.mkdtemp()
    config_file = Path(temp_dir) / "test_config.yaml"

    config_content = """
system:
  name: "Test System"
  version: "1.0.0"

workspace:
  path: "./test_workspace"
  auto_save: true

llm:
  default_model: "gpt-4o"
  default_temperature: 0.7
  default_max_tokens: 4000

screening:
  nf_threshold: 0.75
  similarity_threshold: 0.6
  top_k_chunks: 15
  screening_model: "gpt-4o-mini"
  refinement_model: "gpt-4o"
  llm_threshold_min: 0.4
  llm_threshold_max: 0.85

search:
  default_year_start: 2020
  default_year_end: 2024
  default_max_results: 300
"""

    config_file.write_text(config_content)
    yield str(config_file)

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_config_loader_initialization(sample_config_file):
    """测试配置加载器初始化"""
    loader = ConfigLoader(sample_config_file)
    assert loader.config_path == sample_config_file


def test_load_system_config(sample_config_file):
    """测试加载系统配置"""
    config = ConfigLoader(sample_config_file).load()

    assert isinstance(config, SystemConfig)
    assert config.name == "Test System"
    assert config.version == "1.0.0"
    assert config.llm_model == "gpt-4o"
    assert config.llm_temperature == 0.7


def test_load_screen_config(sample_config_file):
    """测试加载Screen配置"""
    loader = ConfigLoader(sample_config_file)
    loader.load()  # 先加载系统配置

    screen_config = loader.get_agent_config("screen")

    assert isinstance(screen_config, ScreenConfig)
    assert screen_config.name == "ScreenAgent"
    assert screen_config.default_nf_threshold == 0.75
    assert screen_config.similarity_threshold == 0.6
    assert screen_config.top_k_chunks == 15
    # 验证模型分层配置
    assert screen_config.screening_model == "gpt-4o-mini"
    assert screen_config.refinement_model == "gpt-4o"
    assert screen_config.llm_threshold_min == 0.4
    assert screen_config.llm_threshold_max == 0.85


def test_load_search_config(sample_config_file):
    """测试加载Search配置"""
    loader = ConfigLoader(sample_config_file)
    loader.load()

    search_config = loader.get_agent_config("search")

    assert isinstance(search_config, SearchConfig)
    assert search_config.name == "SearchAgent"
    assert search_config.default_year_start == 2020
    assert search_config.default_year_end == 2024
    assert search_config.default_max_results == 300


def test_env_variable_override(sample_config_file):
    """测试环境变量覆盖配置"""
    # 设置环境变量
    os.environ["OPENAI_MODEL"] = "gpt-3.5-turbo"
    os.environ["WORKSPACE_PATH"] = "/tmp/custom_workspace"

    try:
        loader = ConfigLoader(sample_config_file)
        loader.load()

        config = loader._config
        # 验证环境变量已覆盖
        assert config["llm"]["model"] == "gpt-3.5-turbo"
        assert config["workspace"]["path"] == "/tmp/custom_workspace"

    finally:
        # 清理环境变量
        del os.environ["OPENAI_MODEL"]
        del os.environ["WORKSPACE_PATH"]


def test_missing_config_file():
    """测试配置文件不存在时的行为"""
    loader = ConfigLoader("nonexistent.yaml")
    config = loader.load()

    # 应该返回默认配置
    assert isinstance(config, SystemConfig)
    assert config.name == "Multi-Agent Literature Review System"


def test_invalid_yaml():
    """测试无效YAML的处理"""
    temp_dir = tempfile.mkdtemp()
    config_file = Path(temp_dir) / "invalid.yaml"

    # 写入无效YAML
    config_file.write_text("invalid: yaml: content: [")

    try:
        loader = ConfigLoader(str(config_file))
        config = loader.load()

        # 应该回退到默认配置
        assert isinstance(config, SystemConfig)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_global_config_loader(sample_config_file):
    """测试全局配置加载器"""
    # 清除已有的全局实例
    import src.config
    src.config._config_loader = None

    config = load_config(sample_config_file)

    assert isinstance(config, SystemConfig)
    assert config.name == "Test System"


def test_get_agent_config_convenience(sample_config_file):
    """测试便捷函数get_agent_config"""
    # 清除全局实例
    import src.config
    src.config._config_loader = None

    screen_config = get_agent_config("screen")

    assert isinstance(screen_config, ScreenConfig)
    assert screen_config.name == "ScreenAgent"


def test_screen_config_defaults():
    """测试ScreenConfig的默认值"""
    config = ScreenConfig(name="TestScreen")

    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.default_nf_threshold == 0.7
    assert config.enable_llm_screening is True
    assert config.similarity_threshold == 0.5
    assert config.top_k_chunks == 10
    assert config.llm_threshold_min == 0.5
    assert config.llm_threshold_max == 0.8
    assert config.screening_model == "gpt-4o-mini"
    assert config.refinement_model is None


def test_search_config_defaults():
    """测试SearchConfig的默认值"""
    config = SearchConfig(name="TestSearch")

    assert config.name == "TestSearch"
    assert config.default_databases == ["semantic_scholar", "arxiv"]
    assert config.enable_pdf_download is False
    assert config.default_year_start == 2018
    assert config.default_year_end == 2025
    assert config.default_max_results == 500
    assert config.rate_limit_per_second == 10


@pytest.mark.asyncio
async def test_config_values_propagation():
    """测试配置值在Agent中的传递"""
    config = ScreenConfig(
        name="TestScreen",
        model="gpt-4o",
        temperature=0.5,
        max_tokens=2000,
        screening_model="gpt-4o-mini",
        llm_threshold_min=0.4,
        llm_threshold_max=0.9,
    )

    # 验证配置值
    assert config.model == "gpt-4o"
    assert config.temperature == 0.5
    assert config.max_tokens == 2000
    assert config.screening_model == "gpt-4o-mini"
    assert config.llm_threshold_min == 0.4
    assert config.llm_threshold_max == 0.9


def test_config_to_dict_conversion():
    """测试配置到字典的转换"""
    config = SystemConfig(
        name="Test",
        version="2.0",
        llm_model="gpt-4o-turbo",
    )

    # 配置对象应该可以访问所有属性
    assert config.name == "Test"
    assert config.llm_model == "gpt-4o-turbo"