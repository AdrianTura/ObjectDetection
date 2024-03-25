# Setup openvino=2019.3 on ubuntu 20.04

cd /opt/intel
sudo git clone https://github.com/openvinotoolkit/openvino
cd openvino
sudo git checkout 2019_R3
pip install -r /opt/intel/openvino/model-optimizer/requirements.txt
bash /opt/intel/openvino/inference-engine/install_dependencies.sh
cd /opt/intel/openvino/
sudo chmod -R a+rwX /opt/intel/openvino/model-optimizer


# Steps to create the conda environment:

# conda create -y -n model_converter_onnx_to_openvino_2019.3 python=3.6 -c conda-forge
# conda activate model_converter_onnx_to_openvino_2019.3
# pip install pip==9.0.1 
# pip install onnx==1.1.2
# pip install networkx==2.3
# python3 -m pip install coverage m2r pyenchant pylint Sphinx safety test-generator defusedxml

# Steps to activate/dezactivate/delete conda environment:
# conda activate model_converter_torch_to_onnx_p3.6
# conda deactivate
# conda env remove --name model_converter_onnx_to_openvino_2019.3

 python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model /workspace/bsc_thesis/models/mb1-ssd-lite/mb1-ssd-lite.onnx --input_shape [1,3,300,300] --output=Concat_156,Softmax_132 --framework onnx --generate_deprecated_IR_V7 --data_type=FP16 --output_dir models/