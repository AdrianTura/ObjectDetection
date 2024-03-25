# Script that installs openvino on Raspberry Pi
# host OS = 32 bit "RASPBERRY PI OS(LEGACY)"

#Download OpenVino 2019.3/2020.1/2020.4 and pretrained models for test
mkdir models 
cd models

# wget https://download.01.org/opencv/2019/openvinotoolkit/R3/l_openvino_toolkit_runtime_raspbian_p_2019.3.334.tgz
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.1/l_openvino_toolkit_runtime_raspbian_p_2020.1.023.tgz
# wget https://download.01.org/opencv/2020/openvinotoolkit/2020.4/l_openvino_toolkit_runtime_raspbian_p_2020.4.287.tgz

# wget --no-check-certificate https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP16/face-detection-adas-0001.bin
# wget --no-check-certificate https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP16/face-detection-adas-0001.xml

#Install cmake
sudo apt install -y cmake

#Create install folder
sudo mkdir /opt/intel
sudo mkdir /opt/intel/openvino

#Extract
# sudo tar -xvzf l_openvino_toolkit_runtime_raspbian_p_2019.3.334.tgz --strip 1 -C /opt/intel/openvino
sudo tar -xvzf l_openvino_toolkit_runtime_raspbian_p_2020.1.023.tgz --strip 1 -C /opt/intel/openvino
# sudo tar -xvzf l_openvino_toolkit_runtime_raspbian_p_2020.4.287.tgz --strip 1 -C /opt/intel/openvino

# sudo rm -rf l_openvino_toolkit_runtime_raspbian_p_2019.3.334.tgz
sudo rm -rf l_openvino_toolkit_runtime_raspbian_p_2020.1.023.tgz 
# sudo rm -rf l_openvino_toolkit_runtime_raspbian_p_2020.4.287.tgz

#Setup openvino
source /opt/intel/openvino/bin/setupvars.sh
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc

#Setup USB rules
sudo usermod -a -G users "$(whoami)"

#Reset the Raspberry Pi
