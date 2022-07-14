# shapes-project
resnet18 model made using a jetson nano and images from https://www.kaggle.com/datasets/smeschke/four-shapes?select=shapes

The model is alright though somewhat inconsistent in its results and utilises VLC Media player and a sdp file to see the live results

To run the model:
1. Download resnet18.onnx(the model) and shapes-run.py
2. Modify the paths in execute.sh using "--confidence-path" and "--output-path"
3. Use the execute.sh to run the program and model (make sure that the rtp address and port are modified to fit your sdp file)
4. Stop the program whenever you are done and then check the path you specified for output files
