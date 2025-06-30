#!/usr/bin/env fish

# 创建无chat template的纯llama模型，对齐sglang的generate行为
# 支持llama3.2的1b和3b型号，以及llama3.1的8b型号

function create_llama32_model
    set model_size $argv[1]
    set model_name "llama3.2:$model_size-pure"
    
    echo "创建 $model_name 模型..."
    
    # 创建临时Modelfile
    echo "FROM llama3.2:$model_size
TEMPLATE \"\"
PARAMETER stop \"\"
PARAMETER stop \"<|start_header_id|>\"
PARAMETER stop \"<|end_header_id|>\"
PARAMETER stop \"<|eot_id|>\"
PARAMETER stop \"USER:\"
PARAMETER stop \"ASSISTANT:\"
PARAMETER temperature 1.0
PARAMETER num_ctx 4096
PARAMETER seed 1234
PARAMETER num_predict 100" > Modelfile
    
    # 创建模型
    ollama create $model_name -f ./Modelfile
    
    # 清理临时文件
    rm Modelfile
    
    echo "$model_name 创建完成"
end

function create_llama31_model
    set model_size $argv[1]
    set model_name "llama3.1:$model_size-pure"
    
    echo "创建 $model_name 模型..."
    
    # 创建临时Modelfile
    echo "FROM llama3.1:$model_size
TEMPLATE \"\"
PARAMETER stop \"\"
PARAMETER stop \"<|start_header_id|>\"
PARAMETER stop \"<|end_header_id|>\"
PARAMETER stop \"<|eot_id|>\"
PARAMETER stop \"USER:\"
PARAMETER stop \"ASSISTANT:\"
PARAMETER temperature 1.0
PARAMETER num_ctx 4096
PARAMETER seed 1234
PARAMETER num_predict 100" > Modelfile
    
    # 创建模型
    ollama create $model_name -f ./Modelfile
    
    # 清理临时文件
    rm Modelfile
    
    echo "$model_name 创建完成"
end

# 创建Llama 3.2的1b和3b模型
for size in 1b 3b
    create_llama32_model $size
end

# 创建Llama 3.1的8b模型
create_llama31_model 8b

echo "完成! 所有纯文本llama模型已创建"
echo "使用方法: "
echo "- ollama run llama3.2:1b-pure \"你的提示\""
echo "- ollama run llama3.2:3b-pure \"你的提示\""
echo "- ollama run llama3.1:8b-pure \"你的提示\""