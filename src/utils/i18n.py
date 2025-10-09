"""
国际化工具模块
提供多语言支持功能
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, Any, Optional


class I18nManager:
    """国际化管理器"""
    
    def __init__(self):
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.current_language = "zh_CN"
        self.supported_languages = ["zh_CN", "en_US"]
        self.default_language = "zh_CN"
        
        # 加载翻译文件
        self._load_translations()
    
    def _load_translations(self):
        """加载翻译文件"""
        translations_dir = Path(__file__).parent.parent / "config" / "translations"
        
        for lang in self.supported_languages:
            translation_file = translations_dir / f"{lang}.json"
            try:
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations[lang] = json.load(f)
            except FileNotFoundError:
                print(f"警告：未找到语言文件 {translation_file}")
                self.translations[lang] = {}
    
    def set_language(self, language: str):
        """设置当前语言"""
        if language in self.supported_languages:
            self.current_language = language
            # 保存到 session state
            st.session_state.language = language
        else:
            print(f"警告：不支持的语言 {language}")
    
    def get_current_language(self) -> str:
        """获取当前语言"""
        # 优先从 session state 获取
        if 'language' in st.session_state:
            self.current_language = st.session_state.language
        return self.current_language
    
    def get_supported_languages(self) -> Dict[str, str]:
        """获取支持的语言列表"""
        return {
            "zh_CN": "简体中文",
            "en_US": "English"
        }
    
    def t(self, key: str, **kwargs) -> str:
        """
        获取翻译文本
        
        Args:
            key: 翻译键（支持嵌套，如 'app.title'）
            **kwargs: 格式化参数
        
        Returns:
            翻译后的文本
        """
        current_lang = self.get_current_language()
        
        # 尝试获取当前语言的翻译
        translation = self._get_nested_value(self.translations.get(current_lang, {}), key)
        
        # 如果当前语言没有翻译，尝试默认语言
        if translation is None and current_lang != self.default_language:
            translation = self._get_nested_value(self.translations.get(self.default_language, {}), key)
        
        # 如果仍然没有翻译，返回键本身
        if translation is None:
            translation = key
        
        # 应用格式化参数
        try:
            if kwargs:
                translation = translation.format(**kwargs)
        except (KeyError, ValueError) as e:
            print(f"翻译格式化错误: {key} - {e}")
        
        return translation
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """获取嵌套字典值"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    def get_language_flag(self, language: str) -> str:
        """获取语言标志图标"""
        flags = {
            "zh_CN": "🇨🇳",
            "en_US": "🇺🇸"
        }
        return flags.get(language, "🌐")


# 全局国际化管理器实例
_i18n_manager = None

def get_i18n_manager() -> I18nManager:
    """获取全局国际化管理器实例"""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager

def t(key: str, **kwargs) -> str:
    """快捷翻译函数"""
    return get_i18n_manager().t(key, **kwargs)

def set_language(language: str):
    """快捷语言设置函数"""
    get_i18n_manager().set_language(language)

def get_current_language() -> str:
    """获取当前语言"""
    return get_i18n_manager().get_current_language()

def get_supported_languages() -> Dict[str, str]:
    """获取支持的语言列表"""
    return get_i18n_manager().get_supported_languages()

def get_language_flag(language: str) -> str:
    """获取语言标志"""
    return get_i18n_manager().get_language_flag(language)
