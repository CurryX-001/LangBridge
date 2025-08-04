# LLaVA-NeXT Installation Instructions

Follow the LLaVA-NeXT installation instructions. Note that we use CUDA 11.8 and flash_attn 2.7.1.post4.

After installation, clone the evaluation repository:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
```

Install lmms-eval following its instructions, then return to the LLaVA-NeXT folder:

```bash
cd ..
pip install -e .
```

## Additional Requirements

```bash
pip install numpy==1.26.0
pip install pycocotools==2.0.7
pip install httpx==0.23.3
pip install protobuf==3.20
```

## Training and Evaluation

For training and evaluation scripts, refer to:

- Pre-training: `scripts/train/pretrain_clip_vocab.sh`
- Fine-tuning: `scripts/train/direct_finetune_clip_vocab.sh`
- Evaluation: `scripts/train/eval.sh`