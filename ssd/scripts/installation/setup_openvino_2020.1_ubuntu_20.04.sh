# Setup openvino=2020.1 on ubuntu

cd /opt/intel
sudo git clone https://github.com/openvinotoolkit/openvino
cd openvino
sudo git checkout 2020.1
pip install -r /opt/intel/openvino/model-optimizer/requirements.txt
bash /opt/intel/openvino/install_dependencies.sh
cd /opt/intel/openvino/
sudo chmod -R a+rwX /opt/intel/openvino/model-optimizer


# Steps to create the conda environment:

# conda create -y -n model_converter_onnx_to_openvino_2020.1 python=3.6 -c conda-forge
# conda activate model_converter_onnx_to_openvino_2020.1
# pip install pip==9.0.1
# python3 -m pip install networkx==2.3 coverage m2r pyenchant pylint Sphinx safety test-generator defusedxml
# pip install onnx==1.1.2

# Steps to activate/dezactivate/delete conda environment:
# conda activate model_converter_onnx_to_openvino_2020.1
# conda deactivate
# conda env remove --name model_converter_onnx_to_openvino_2020.1