# packages/leann-core/src/leann/registry.py

# 全局的后端注册表字典
BACKEND_REGISTRY = {}

def register_backend(name: str):
    """一个用于注册新后端类的装饰器。"""
    def decorator(cls):
        print(f"INFO: Registering backend '{name}'")
        BACKEND_REGISTRY[name] = cls
        return cls
    return decorator