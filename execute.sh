imagenet-camera --model=/shapes_project/models/shapes/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=/shapes_project/data/shapes/labels.txt /dev/video0 --width=640 --height=480 rtp://192.168.5.102:1234