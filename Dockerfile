From python:3.11-slim

COPY .

pip install --no-cache-dir -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install git+https://github.com/microsoft/MoGe.git