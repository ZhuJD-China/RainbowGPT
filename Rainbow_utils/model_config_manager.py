import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatBaichuan
from langchain_core.messages import HumanMessage

@dataclass
class ModelConfig:
    model_name: str
    api_base: str
    api_key: str
    temperature: float = 0.0
    
    def get_masked_key(self):
        """返回掩码处理后的API key"""
        if not self.api_key:
            return ""
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return self.api_key[:4] + "*" * (len(self.api_key) - 8) + self.api_key[-4:]

class ModelConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.baichuan_api_key = os.getenv('BAICHUAN_API_KEY', '')
        self.qwen_api_key = os.getenv('DASHSCOPE_API_KEY', '')
        self.default_api_base = "https://api.chatanywhere.tech"
        
        # 默认配置
        self.gpt_config = ModelConfig(
            model_name="gpt-4",
            api_base=self.default_api_base,
            api_key=self.openai_api_key
        )
        
        self.private_llm_config = ModelConfig(
            model_name="gpt-4-mini",
            api_base=self.default_api_base,
            api_key=""
        )

        # Add Baichuan config
        self.baichuan_config = ModelConfig(
            model_name="Baichuan3-Turbo-128k",
            api_base="",  # Baichuan doesn't need api_base
            api_key=self.baichuan_api_key
        )
        
        # Add Qwen config
        self.qwen_config = ModelConfig(
            model_name="qwen-long",
            api_base="",
            api_key=self.qwen_api_key
        )
        
        self.active_config = self.gpt_config
    
    def set_gpt_config(self, model_name: str, api_base: str = None, temperature: float = 0.0):
        """设置GPT模型配置"""
        self.gpt_config.model_name = model_name
        if api_base:  # 添加对api_base的更新
            self.gpt_config.api_base = api_base
        self.gpt_config.temperature = temperature
        # 如果当前是GPT配置，更新活动配置
        if self.active_config == self.gpt_config:
            self.active_config = self.gpt_config
    
    def set_private_llm_config(self, model_name: str, api_base: str, api_key: str, temperature: float = 0.0):
        """设置私有模型配置"""
        # 创建新的配置而不是修改现有的
        self.private_llm_config = ModelConfig(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature
        )
        # 如果当前是私有配置，更新活动配置
        if self.active_config != self.gpt_config:
            self.active_config = self.private_llm_config
    
    def use_gpt_model(self):
        """切换到GPT模型"""
        self.active_config = self.gpt_config
        # 确保使用最新的API key
        if not self.active_config.api_key:
            self.active_config.api_key = self.openai_api_key
    
    def use_private_llm_model(self):
        """切换到私有模型"""
        self.active_config = self.private_llm_config
    
    def update_active_config(self, config: ModelConfig):
        """更新当前活动的配置"""
        self.active_config = config
    
    def get_active_config(self) -> ModelConfig:
        """获取当前活动的配置"""
        return self.active_config
    
    def get_masked_api_key(self) -> str:
        """返回当前配置的掩码处理后的API key"""
        return self.active_config.get_masked_key() 
    
    def use_baichuan_model(self):
        """Switch to Baichuan model"""
        self.active_config = self.baichuan_config
    
    def use_qwen_model(self):
        """Switch to Qwen model"""
        self.active_config = self.qwen_config
