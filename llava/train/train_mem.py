from llava.train.train import train
# import os
# os.environ['MASTER_PORT'] = '32423'
import os

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
