import os
import sys
import time
import subprocess

if __name__ == "__main__":
    # detection
    subprocess.call("CUDA_VISIBLE_DEVICES=0 python -u ./table_recognition/table_inference.py 8 0 0 &"
                    "CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 8 1 0 &"
                    "CUDA_VISIBLE_DEVICES=2 python -u ./table_recognition/table_inference.py 8 2 0 &"
                    "CUDA_VISIBLE_DEVICES=3 python -u ./table_recognition/table_inference.py 8 3 0 &"
                    "CUDA_VISIBLE_DEVICES=4 python -u ./table_recognition/table_inference.py 8 4 0 &"
                    "CUDA_VISIBLE_DEVICES=5 python -u ./table_recognition/table_inference.py 8 5 0 &"
                    "CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 8 6 0 &"
                    "CUDA_VISIBLE_DEVICES=7 python -u ./table_recognition/table_inference.py 8 7 0", shell=True)
    time.sleep(60)

    # structure
    subprocess.call("CUDA_VISIBLE_DEVICES=0 python -u ./table_recognition/table_inference.py 8 0 2 &"
                    "CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 8 1 2 &"
                    "CUDA_VISIBLE_DEVICES=2 python -u ./table_recognition/table_inference.py 8 2 2 &"
                    "CUDA_VISIBLE_DEVICES=3 python -u ./table_recognition/table_inference.py 8 3 2 &"
                    "CUDA_VISIBLE_DEVICES=4 python -u ./table_recognition/table_inference.py 8 4 2 &"
                    "CUDA_VISIBLE_DEVICES=5 python -u ./table_recognition/table_inference.py 8 5 2 &"
                    "CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 8 6 2 &"
                    "CUDA_VISIBLE_DEVICES=7 python -u ./table_recognition/table_inference.py 8 7 2", shell=True)
    time.sleep(60)

    # recognition
    subprocess.call("CUDA_VISIBLE_DEVICES=0 python -u ./table_recognition/table_inference.py 8 1 1", shell=True)