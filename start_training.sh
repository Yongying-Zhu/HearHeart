#!/bin/bash
# ========================================
# HearHeart 数据整理与训练完整脚本
# 数据下载完成后执行此脚本
# ========================================

set -e

echo "========================================"
echo "HearHeart 数据整理与训练"
echo "========================================"
echo ""

# 激活虚拟环境检查
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ 请先激活虚拟环境:"
    echo "   source ~/anaconda3/envs/AnYujin/bin/activate"
    exit 1
fi

echo "✓ 虚拟环境: $CONDA_DEFAULT_ENV"
echo ""

# 进入项目目录
cd ~/hearheart

# ========================================
# 第一步: 整理下载的数据
# ========================================
echo "========================================"
echo "第 1 步: 整理下载的数据"
echo "========================================"
echo ""

# 检查下载目录
if [ -d "./physionet.org" ]; then
    DOWNLOADED_FILES=$(find ./physionet.org -type f 2>/dev/null | wc -l)
    echo "已下载文件数: $DOWNLOADED_FILES"

    if [ "$DOWNLOADED_FILES" -eq 0 ]; then
        echo "❌ 下载目录为空，请确保数据已下载完成"
        exit 1
    fi

    echo ""
    echo "正在移动文件到 training_data 目录..."

    # 创建目录
    mkdir -p training_data

    # 移动所有数据文件
    find ./physionet.org/files/circor-heart-sound/1.0.3/training_data/ -type f \
        \( -name "*.txt" -o -name "*.wav" -o -name "*.tsv" -o -name "*.hea" \) \
        -exec cp {} ./training_data/ \; 2>/dev/null || true

    echo "✓ 文件移动完成"
    echo ""

    # 清理临时目录
    echo "清理临时下载目录..."
    rm -rf ./physionet.org
    echo "✓ 清理完成"
    echo ""
else
    echo "未找到下载目录，检查 training_data 是否已有数据..."
fi

# 验证数据
echo "========================================"
echo "数据统计"
echo "========================================"

TXT_COUNT=$(find training_data -name "*.txt" 2>/dev/null | wc -l)
WAV_COUNT=$(find training_data -name "*.wav" 2>/dev/null | wc -l)
TSV_COUNT=$(find training_data -name "*.tsv" 2>/dev/null | wc -l)
HEA_COUNT=$(find training_data -name "*.hea" 2>/dev/null | wc -l)

echo "患者记录 (.txt): $TXT_COUNT"
echo "音频文件 (.wav): $WAV_COUNT"
echo "标注文件 (.tsv): $TSV_COUNT"
echo "头文件   (.hea): $HEA_COUNT"
echo ""

if [ "$TXT_COUNT" -eq 0 ] || [ "$WAV_COUNT" -eq 0 ]; then
    echo "❌ 训练数据不足或缺失"
    echo ""
    echo "请确保:"
    echo "1. 数据已完全下载"
    echo "2. 数据文件在 ~/hearheart/training_data/ 目录中"
    exit 1
fi

echo "✓ 数据验证通过！"
echo ""

# ========================================
# 第二步: 检查依赖
# ========================================
echo "========================================"
echo "第 2 步: 检查依赖包"
echo "========================================"
echo ""

# 检查 librosa
if python -c "import librosa" 2>/dev/null; then
    LIBROSA_VER=$(python -c "import librosa; print(librosa.__version__)")
    echo "✓ librosa $LIBROSA_VER"
else
    echo "正在安装 librosa..."
    pip install -q librosa>=0.10.0
    echo "✓ librosa 安装完成"
fi

# 检查 PyTorch
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")
echo "✓ PyTorch $TORCH_VER"

# 检查 CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✓ CUDA $CUDA_VER"
    echo "✓ GPU: $GPU_NAME"
else
    echo "⚠️  CUDA 不可用，将使用 CPU 训练（速度较慢）"
fi

echo ""

# ========================================
# 第三步: 创建必要目录
# ========================================
echo "========================================"
echo "第 3 步: 准备环境"
echo "========================================"
echo ""

mkdir -p test_data model test_outputs

echo "✓ 目录结构就绪"
ls -ld training_data test_data model test_outputs
echo ""

# ========================================
# 第四步: 开始训练
# ========================================
echo "========================================"
echo "第 4 步: 开始训练模型"
echo "========================================"
echo ""
echo "⏱️  训练时间预估: 2-6 小时（取决于硬件配置）"
echo ""
echo "训练数据: $TXT_COUNT 个患者, $WAV_COUNT 个音频文件"
echo ""
read -p "按 Enter 键开始训练，或按 Ctrl+C 取消... " -t 10 || echo ""

echo ""
echo "🚀 训练开始..."
echo "----------------------------------------"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 执行训练
python train_model.py training_data model 2

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "----------------------------------------"
echo "✓ 训练完成！"
echo "⏱️  耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""

# ========================================
# 第五步: 查看结果
# ========================================
echo "========================================"
echo "第 5 步: 训练结果"
echo "========================================"
echo ""

if [ -d "model" ]; then
    echo "📁 模型文件:"
    ls -lh model/
    echo ""

    MODEL_SIZE=$(du -sh model/ | cut -f1)
    echo "📊 模型大小: $MODEL_SIZE"
    echo ""
fi

# ========================================
# 完成
# ========================================
echo "========================================"
echo "✓ 全部完成！"
echo "========================================"
echo ""
echo "📂 项目位置: ~/hearheart"
echo "🤖 训练模型: ~/hearheart/model/"
echo ""
echo "后续操作:"
echo ""
echo "1️⃣  测试模型（如有测试数据）:"
echo "   python run_model.py model test_data test_outputs 2"
echo ""
echo "2️⃣  评估模型:"
echo "   python evaluate_model.py test_data test_outputs scores.csv class_scores.csv"
echo ""
echo "3️⃣  查看模型文件:"
echo "   ls -lh model/"
echo ""
