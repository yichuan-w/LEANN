# Manylinux Build Strategy

## 问题概述

Colab和许多Linux环境需要`manylinux`兼容的wheels，而不是特定于某个Linux发行版的wheels。我们之前的尝试过于复杂，试图手动处理所有的依赖关系。

## 新策略

基于社区最佳实践和其他成功项目的经验，我们采用了以下简化方案：

### 1. 使用 cibuildwheel

`cibuildwheel`是PyPA推荐的构建多平台wheels的工具，它：
- 自动处理manylinux环境设置
- 管理不同Python版本的构建
- 提供标准化的测试框架

### 2. 使用 manylinux_2_28

- 从`manylinux2014`升级到`manylinux_2_28`
- 提供更现代的基础库
- 使用`dnf`而不是`yum`
- 更好地支持现代Python版本

### 3. CMake配置

在`pyproject.toml`中添加：
```toml
[tool.scikit-build.cmake.define]
Python_FIND_VIRTUALENV = "ONLY"
Python3_FIND_VIRTUALENV = "ONLY"
```

这帮助CMake在cibuildwheel的虚拟环境中正确找到Python。

### 4. 简化的依赖安装

只安装必要的系统依赖：
- `gcc-c++`
- `boost-devel`
- `zeromq-devel`
- `openblas-devel`
- `cmake`

## 测试策略

1. **手动触发**: 使用`workflow_dispatch`在GitHub Actions上测试
2. **自动测试**: 在`fix/manylinux-*`分支上推送时自动运行
3. **PR测试**: 当PR修改相关文件时自动测试

## 如何使用

### 在本地测试
```bash
# 安装cibuildwheel
pip install cibuildwheel

# 构建wheels
cibuildwheel --platform linux packages/leann-backend-hnsw
```

### 在GitHub Actions测试
1. 推送到`fix/manylinux-compatibility`分支
2. 或手动触发"Test Manylinux Build"工作流

## 下一步

1. **监控CI运行结果**
2. **根据错误调整依赖**
3. **测试生成的wheels在Colab的兼容性**
4. **如果成功，将更改合并到main**

## 参考

- [cibuildwheel文档](https://cibuildwheel.readthedocs.io/)
- [manylinux规范](https://github.com/pypa/manylinux)
- [scikit-build-core文档](https://scikit-build-core.readthedocs.io/) 