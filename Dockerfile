FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04
RUN pip install --upgrade 'transformers>=4.39.0'
RUN pip install accelerate \
    && pip install peft bitsandbytes \
    && pip install --upgrade 'trl>=0.8.3'
