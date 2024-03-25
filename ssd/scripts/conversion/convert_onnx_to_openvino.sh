sudo make docker-start-interactive

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model models/mb1-ssd/mb1-ssd.onnx --input_shape [1,3,300,300] --framework onnx --generate_deprecated_IR_V7 --data_type=FP16