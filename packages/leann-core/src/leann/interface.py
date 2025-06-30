from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class LeannBackendBuilderInterface(ABC):
    """用于构建索引的后端接口"""
    
    @abstractmethod 
    def build(self, data: np.ndarray, index_path: str, **kwargs) -> None:
        """构建索引
        
        Args:
            data: 向量数据 (N, D)
            index_path: 索引保存路径
            **kwargs: 后端特定的构建参数
        """
        pass

class LeannBackendSearcherInterface(ABC):
    """用于搜索的后端接口"""
    
    @abstractmethod
    def __init__(self, index_path: str, **kwargs):
        """初始化搜索器
        
        Args:
            index_path: 索引文件路径
            **kwargs: 后端特定的加载参数
        """
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, Any]:
        """搜索最近邻
        
        Args:
            query: 查询向量 (1, D) 或 (B, D)
            top_k: 返回的最近邻数量
            **kwargs: 搜索参数
            
        Returns:
            {"labels": [...], "distances": [...]}
        """
        pass

class LeannBackendFactoryInterface(ABC):
    """后端工厂接口"""
    
    @staticmethod
    @abstractmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        """创建 Builder 实例"""
        pass
    
    @staticmethod
    @abstractmethod  
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        """创建 Searcher 实例"""
        pass