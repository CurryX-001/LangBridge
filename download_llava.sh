#!/bin/bash

# 设置根目录
sudo apt-get install -y wget unzip
ROOT_DIR="/scratch/Codebook"

# 创建必要的目录结构
mkdir -p ${ROOT_DIR}
mkdir -p ${ROOT_DIR}/playground/data/{pretrain,coco/train2017,gqa/images,ocr_vqa/images,textvqa/train_images,vg/{VG_100K,VG_100K_2}}

# 进入根目录
cd ${ROOT_DIR}

检查和下载函数
download_if_not_exists() {
    local url=$1
    local output=$2
    local desc=$3
    
    if [ -f "$output" ]; then
        echo "文件已存在: $desc"
        read -p "是否重新下载？(y/n) " choice
        case "$choice" in 
            y|Y ) wget --progress=bar:force:noscroll "$url" -O "$output";;
            * ) echo "跳过下载 $desc";;
        esac
    else
        echo "开始下载: $desc"
        wget --progress=bar:force:noscroll "$url" -O "$output"
    fi
}

# 第一阶段：下载所有文件
echo "开始下载所有数据集..."

echo "检查预训练数据集..."
download_if_not_exists "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json" \
    "playground/data/pretrain/blip_laion_cc_sbu_558k.json" "预训练数据集标注文件"
download_if_not_exists "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip" \
    "playground/data/pretrain/images.zip" "预训练数据集图片"

echo "检查微调数据集标注文件..."
download_if_not_exists "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json" \
    "playground/data/llava_v1_5_mix665k.json" "微调数据集标注文件"

echo "检查COCO数据集..."
download_if_not_exists "http://images.cocodataset.org/zips/train2017.zip" \
    "train2017.zip" "COCO数据集"

# 下载GQA数据集
echo "检查GQA数据集..."
download_if_not_exists "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    "images.zip" "GQA数据集"

echo "检查TextVQA数据集..."
download_if_not_exists "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
    "train_val_images.zip" "TextVQA数据集"

echo "检查Visual Genome数据集..."
download_if_not_exists "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" \
    "vg_part1.zip" "Visual Genome数据集部分1"
download_if_not_exists "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
    "vg_part2.zip" "Visual Genome数据集部分2"

# 第二阶段：解压文件
echo "开始解压数据集..."

# 解压函数
extract_if_needed() {
    local zip_file=$1
    local extract_dir=$2
    local desc=$3
    
    # 检查目标目录是否为空
    if [ -d "$extract_dir" ] && [ "$(ls -A $extract_dir)" ]; then
        echo "$desc 目录非空，可能已经解压过"
        read -p "是否重新解压？(y/n) " choice
        case "$choice" in 
            y|Y )
                echo "解压 $desc..."
                unzip -q "$zip_file" -d "$extract_dir"
                rm -f "$zip_file"
                ;;
            * ) echo "跳过解压 $desc";;
        esac
    else
        echo "解压 $desc..."
        unzip -q "$zip_file" -d "$extract_dir"
        rm -f "$zip_file"
    fi
}

# 解压所有数据集
[ -f "playground/data/pretrain/images.zip" ] && extract_if_needed "playground/data/pretrain/images.zip" "playground/data/pretrain" "预训练数据集"
[ -f "train2017.zip" ] && extract_if_needed "train2017.zip" "playground/data/coco" "COCO数据集"
[ -f "images.zip" ] && extract_if_needed "images.zip" "playground/data/gqa" "GQA数据集"
[ -f "train_val_images.zip" ] && extract_if_needed "train_val_images.zip" "playground/data/textvqa" "TextVQA数据集"
[ -f "vg_part1.zip" ] && extract_if_needed "vg_part1.zip" "playground/data/vg/" "Visual Genome部分1"
[ -f "vg_part2.zip" ] && extract_if_needed "vg_part2.zip" "playground/data/vg/" "Visual Genome部分2"

echo "所有数据集处理完成！"
echo "注意：OCR-VQA数据集需要手动从Google Drive下载。"
echo "请从以下链接下载OCR-VQA：https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_"
echo "下载后请将图片放置在：${ROOT_DIR}/playground/data/ocr_vqa/images/"

# 显示最终的目录结构
echo "最终目录结构："
tree ${ROOT_DIR}/playground/data -L 2