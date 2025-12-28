#!/bin/bash
# ========================================
# HearHeart 项目训练与测试脚本
# ========================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "HearHeart 训练测试脚本"
echo "========================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "当前工作目录: $SCRIPT_DIR"
echo ""

# 检查是否在虚拟环境中
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  警告: 未检测到虚拟环境"
    echo "请先激活虚拟环境: source ~/anaconda3/envs/AnYujin/bin/activate"
    exit 1
fi

echo "✓ 虚拟环境: $CONDA_DEFAULT_ENV"
echo ""

# 安装依赖
echo "========================================"
echo "1. 检查并安装依赖包"
echo "========================================"

# 检查librosa是否已安装
if python -c "import librosa" 2>/dev/null; then
    LIBROSA_VERSION=$(python -c "import librosa; print(librosa.__version__)")
    echo "✓ librosa 已安装 (版本: $LIBROSA_VERSION)"
else
    echo "正在安装 librosa..."
    pip install librosa>=0.10.0 -q
    echo "✓ librosa 安装完成"
fi

echo ""

# 创建目录结构
echo "========================================"
echo "2. 创建目录结构"
echo "========================================"

mkdir -p training_data
mkdir -p test_data
mkdir -p model
mkdir -p test_outputs

echo "✓ 目录结构已创建"
echo ""
ls -ld training_data test_data model test_outputs
echo ""

# 检查训练数据
echo "========================================"
echo "3. 检查数据"
echo "========================================"

TRAIN_COUNT=$(find training_data -type f 2>/dev/null | wc -l)
TEST_COUNT=$(find test_data -type f 2>/dev/null | wc -l)

echo "训练数据文件数: $TRAIN_COUNT"
echo "测试数据文件数: $TEST_COUNT"
echo ""

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "❌ 错误: training_data 目录为空"
    echo ""
    echo "请将训练数据放入: $SCRIPT_DIR/training_data"
    echo ""
    echo "数据格式要求: PhysioNet Challenge 2022 格式"
    echo "数据下载: https://physionetchallenges.org/2022/"
    echo ""
    exit 1
fi

echo "✓ 训练数据已就绪"
echo ""

# 训练模型
echo "========================================"
echo "4. 开始训练模型"
echo "========================================"
echo "这可能需要较长时间，请耐心等待..."
echo ""

python train_model.py training_data model 2

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 模型训练完成"
    echo "模型保存位置: $SCRIPT_DIR/model"
    echo ""
    ls -lh model/
    echo ""
else
    echo ""
    echo "❌ 模型训练失败"
    exit 1
fi

# 运行测试
if [ "$TEST_COUNT" -gt 0 ]; then
    echo "========================================"
    echo "5. 运行模型测试"
    echo "========================================"
    echo ""

    python run_model.py model test_data test_outputs 2

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 测试完成"
        echo "输出保存位置: $SCRIPT_DIR/test_outputs"
        echo ""

        # 评估模型
        echo "========================================"
        echo "6. 评估模型性能"
        echo "========================================"
        echo ""

        python evaluate_model.py test_data test_outputs scores.csv class_scores.csv

        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ 评估完成"
            echo ""

            if [ -f "scores.csv" ]; then
                echo "--- 总体评分 (scores.csv) ---"
                cat scores.csv
                echo ""
            fi

            if [ -f "class_scores.csv" ]; then
                echo "--- 分类评分 (class_scores.csv) ---"
                head -20 class_scores.csv
                echo ""
            fi
        fi
    else
        echo ""
        echo "❌ 测试失败"
    fi
else
    echo "========================================"
    echo "5. 跳过测试"
    echo "========================================"
    echo "⚠️  未找到测试数据，跳过测试步骤"
    echo ""
    echo "如需测试，请将测试数据放入: $SCRIPT_DIR/test_data"
    echo "然后运行: python run_model.py model test_data test_outputs 2"
    echo ""
fi

echo "========================================"
echo "✓ 所有步骤完成"
echo "========================================"
echo ""
echo "训练模型: $SCRIPT_DIR/model"
echo "测试输出: $SCRIPT_DIR/test_outputs"
echo ""
