# Convert the models to .onnx

export CUDA_VISIBLE_DEVICES=""

python3 convert_to_caffe2_models.py mb1-ssd results/mb1-ssd/mb1-ssd.pth models/face-detection-labels.txt 
python3 convert_to_caffe2_models.py mb1-ssd-lite results/mb1-ssd-lite/mb1-ssd-lite.pth models/face-detection-labels.txt 
# python convert_to_caffe2_models.py mb2-ssd-lite results/mb2-ssd-lite/mb2-ssd-lite.pth models/face-detection-labels.txt 
# python convert_to_caffe2_models.py mb3-large-ssd-lite results/mb3-large-ssd-lite/mb3-large-ssd-lite.pth models/face-detection-labels.txt
# python convert_to_caffe2_models.py mb3-small-ssd-lite results/mb3-small-ssd-lite/mb3-small-ssd-lite.pth models/face-detection-labels.txt
# python convert_to_caffe2_models.py msq-ssd-lite results/msq-ssd-lite/msq-ssd-lite.pth models/face-detection-labels.txt
python3 convert_to_caffe2_models.py vgg16-ssd results/vgg16-ssd/vgg16-ssd.pth models/face-detection-labels.txt


# Steps to create the conda environment:

# conda create -y -n model_converter_torch_to_onnx python=3.6 -c conda-forge
# pip install torch numpy torchvision opencv-python future
# pip install pip==9.0.1 
# pip install --upgrade google-api-python-client
# conda install -c conda-forge onnx
# pip uninstall protobuf
# pip install protobuf
# Steps to activate/deactivate conda environment:
# conda activate model_converter_torch_to_onnx
# conda deactivate
# conda env remove --name model_converter_torch_to_onnx