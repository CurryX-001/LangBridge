#!/bin/bash

# Set root directory
sudo apt-get install -y wget unzip
ROOT_DIR=""

# Create necessary directory structure
mkdir -p ${ROOT_DIR}
mkdir -p ${ROOT_DIR}/playground/data/{pretrain,coco/train2017,gqa/images,ocr_vqa/images,textvqa/train_images,vg/{VG_100K,VG_100K_2}}

# Enter root directory
cd ${ROOT_DIR}

# Check and download function
download_if_not_exists() {
    local url=$1
    local output=$2
    local desc=$3
    
    if [ -f "$output" ]; then
        echo "File already exists: $desc"
        read -p "Do you want to re-download? (y/n) " choice
        case "$choice" in 
            y|Y ) wget --progress=bar:force:noscroll "$url" -O "$output";;
            * ) echo "Skip downloading $desc";;
        esac
    else
        echo "Start downloading: $desc"
        wget --progress=bar:force:noscroll "$url" -O "$output"
    fi
}

# Phase 1: Download all files
echo "Start downloading all datasets..."

echo "Checking pre-training datasets..."
download_if_not_exists "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json" \
    "playground/data/pretrain/blip_laion_cc_sbu_558k.json" "Pre-training dataset annotation file"
download_if_not_exists "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip" \
    "playground/data/pretrain/images.zip" "Pre-training dataset images"

echo "Checking fine-tuning dataset annotation files..."
download_if_not_exists "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json" \
    "playground/data/llava_v1_5_mix665k.json" "Fine-tuning dataset annotation file"

echo "Checking COCO dataset..."
download_if_not_exists "http://images.cocodataset.org/zips/train2017.zip" \
    "train2017.zip" "COCO dataset"

# Download GQA dataset
echo "Checking GQA dataset..."
download_if_not_exists "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    "images.zip" "GQA dataset"

echo "Checking TextVQA dataset..."
download_if_not_exists "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
    "train_val_images.zip" "TextVQA dataset"

echo "Checking Visual Genome dataset..."
download_if_not_exists "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" \
    "vg_part1.zip" "Visual Genome dataset part 1"
download_if_not_exists "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
    "vg_part2.zip" "Visual Genome dataset part 2"

# Phase 2: Extract files
echo "Start extracting datasets..."

# Extract function
extract_if_needed() {
    local zip_file=$1
    local extract_dir=$2
    local desc=$3
    
    # Check if target directory is empty
    if [ -d "$extract_dir" ] && [ "$(ls -A $extract_dir)" ]; then
        echo "$desc directory is not empty, may have been extracted already"
        read -p "Do you want to re-extract? (y/n) " choice
        case "$choice" in 
            y|Y )
                echo "Extracting $desc..."
                unzip -q "$zip_file" -d "$extract_dir"
                rm -f "$zip_file"
                ;;
            * ) echo "Skip extracting $desc";;
        esac
    else
        echo "Extracting $desc..."
        unzip -q "$zip_file" -d "$extract_dir"
        rm -f "$zip_file"
    fi
}

# Extract all datasets
[ -f "playground/data/pretrain/images.zip" ] && extract_if_needed "playground/data/pretrain/images.zip" "playground/data/pretrain" "Pre-training dataset"
[ -f "train2017.zip" ] && extract_if_needed "train2017.zip" "playground/data/coco" "COCO dataset"
[ -f "images.zip" ] && extract_if_needed "images.zip" "playground/data/gqa" "GQA dataset"
[ -f "train_val_images.zip" ] && extract_if_needed "train_val_images.zip" "playground/data/textvqa" "TextVQA dataset"
[ -f "vg_part1.zip" ] && extract_if_needed "vg_part1.zip" "playground/data/vg/" "Visual Genome part 1"
[ -f "vg_part2.zip" ] && extract_if_needed "vg_part2.zip" "playground/data/vg/" "Visual Genome part 2"

# Phase 3: Download evaluation datasets
echo "Start downloading evaluation datasets..."

# Create evaluation data directory
mkdir -p ${ROOT_DIR}/playground/data/eval

echo "Downloading LLaVA evaluation dataset split files..."
BASE_URL="https://huggingface.co/datasets/ainbo/llava_eval_dataset/resolve/main"

# Download all split files (part_aa, part_ab, etc.)
for part in aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as at au av aw ax ay az ba bb bc bd be bf bg bh bi bj bk bl bm bn bo bp bq br bs bt bu bv bw bx by bz ca cb cc cd ce cf cg ch ci cj ck cl cm cn co cp cq cr cs ct cu cv cw cx cy cz; do
    part_file="part_${part}"
    if wget --spider "${BASE_URL}/${part_file}" 2>/dev/null; then
        download_if_not_exists "${BASE_URL}/${part_file}" \
            "${part_file}" "Evaluation dataset split ${part_file}"
    else
        echo "Split ${part_file} does not exist, stopping download of subsequent splits"
        break
    fi
done

echo "Merging split files into eval.zip..."
# Check if split files exist
if ls part_* 1> /dev/null 2>&1; then
    cat part_* > eval.zip
    echo "Split files merged successfully"
    
    # Delete split files
    rm -f part_*
    echo "Split files cleanup completed"
else
    echo "Error: Split files not found"
    exit 1
fi

echo "Extracting eval.zip to ./playground/data/eval/..."
if [ -f "eval.zip" ]; then
    unzip -o eval.zip -d "${ROOT_DIR}/playground/data/eval/"
    echo "eval.zip extraction completed"
    
    # Delete eval.zip file
    rm -f eval.zip
    echo "eval.zip cleanup completed"
else
    echo "Error: eval.zip file does not exist"
    exit 1
fi

echo "All datasets processing completed!"
echo "Note: OCR-VQA dataset needs to be downloaded manually from Google Drive."
echo "Please download OCR-VQA from the following link: https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain"
echo "After downloading, please place images in: ${ROOT_DIR}/playground/data/ocr_vqa/images/"

# Display final directory structure
echo "Final directory structure:"
echo "Evaluation data directory:"
if command -v tree >/dev/null 2>&1; then
    tree ${ROOT_DIR}/playground/data -L 3
else
    echo "Install tree command to view complete directory structure: brew install tree (macOS) or apt-get install tree (Linux)"
    find ${ROOT_DIR}/playground/data -type d -maxdepth 3 | sort
fi