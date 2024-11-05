from llava.train.train import train
# import os
# os.environ['MASTER_PORT'] = '32423'
import os

# 设置环境变量
os.environ['NCCL_IB_HCA'] = 'mlx5_2,mlx5_3'
os.environ['NCCL_SOCKET_IFNAME']='bond0'
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
