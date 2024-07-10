FROM 763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-training:2.1-transformers4.36-gpu-py310-cu121-ubuntu20.04
RUN pip install --upgrade 'transformers>=4.39.0'
RUN pip install accelerate==0.32.1 \
    && pip install peft==0.11.1 bitsandbytes==0.43.1 \
    && pip install trl==0.8.3
