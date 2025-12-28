#!/bin/bash
# ========================================
# 数据整理脚本
# ========================================

set -e

cd ~/hearheart

echo "========================================"
echo "整理下载的数据文件"
echo "========================================"
echo ""

# 检查下载的文件
echo "步骤 1/3: 检查下载的文件"
DOWNLOADED_FILES=$(find ./physionet.org -type f 2>/dev/null | wc -l)
echo "已下载文件数: $DOWNLOADED_FILES"

if [ "$DOWNLOADED_FILES" -eq 0 ]; then
    echo "❌ 未找到下载的文件"
    echo "请确保 wget 下载已完成"
    exit 1
fi

echo "✓ 文件下载完成"
echo ""

# 移动文件到 training_data
echo "步骤 2/3: 移动文件到 training_data 目录"
mkdir -p training_data

# 复制所有数据文件
find ./physionet.org/files/circor-heart-sound/1.0.3/training_data/ -type f \
    \( -name "*.txt" -o -name "*.wav" -o -name "*.tsv" -o -name "*.hea" \) \
    -exec cp {} ./training_data/ \; 2>/dev/null || true

echo "✓ 文件移动完成"
echo ""

# 清理下载的临时目录
echo "步骤 3/3: 清理临时文件"
rm -rf ./physionet.org

echo "✓ 清理完成"
echo ""

# 验证数据
echo "========================================"
echo "数据统计"
echo "========================================"

TXT_COUNT=$(find training_data -name "*.txt" | wc -l)
WAV_COUNT=$(find training_data -name "*.wav" | wc -l)
TSV_COUNT=$(find training_data -name "*.tsv" | wc -l)
HEA_COUNT=$(find training_data -name "*.hea" | wc -l)

echo "患者记录文件 (.txt): $TXT_COUNT"
echo "音频文件 (.wav):     $WAV_COUNT"
echo "标注文件 (.tsv):     $TSV_COUNT"
echo "头文件 (.hea):       $HEA_COUNT"
echo ""

if [ "$TXT_COUNT" -gt 0 ] && [ "$WAV_COUNT" -gt 0 ]; then
    echo "✓ 数据准备就绪！"
    echo ""
    echo "数据位置: $(pwd)/training_data"
    echo ""
    echo "下一步: 运行训练脚本"
    echo "  ./run_training.sh"
else
    echo "❌ 数据不完整，请检查下载"
fi

echo ""
