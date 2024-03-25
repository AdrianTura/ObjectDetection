# Script that tests installation of openvino on Raspberry Pi

#Setup current session
sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh

#Create build folder
mkdir build 
cd build

#Build object_detection_sample_ssd
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" /opt/intel/openvino/deployment_tools/inference_engine/samples
make -j2 object_detection_sample_ssd

#Perform inference
cd ..
build/armv7l/Release/object_detection_sample_ssd -m models/face-detection-adas-0001.xml -d MYRIAD -i images/face.jpeg