#!/usr/bin/env python3

import jetson.inference
import jetson.utils

import argparse
import sys


# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")
parser.add_argument("--confidence-file", type=str, default="confidence", help="path to confidence txt file that will be created")
parser.add_argument("--output-file", type=str, default="confidence", help="path to output txt file that will be created")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
font = jetson.utils.cudaFont()
confidence-path = args.confidence-file
output-path = args.output-file
present_label = "New Session"
past_label = "placeholder"

if os.path.isdir(confidence-path):
    a = open(confidence-path + '/confidence.txt','w')
else:
    os.system("mkdir shapes-confidence")
    a = open('shapes-confidence/confidence.txt', 'w')
	
if os.path.isdir(output-path):
    b = open(output-path + '/output.txt', 'w')
else:
    os.system("mkdir shapes-output")
    b = open('shapes-output/output.txt', 'w')
	
# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # classify the image
    class_id, confidence = net.Classify(img)

    # find the object description
    class_desc = net.GetClassDesc(class_id)

    #set present_label to current class_desc with strong enough confidence
    if confidence > 0.5:
	a.write("High Confidence: " + str(confidence)*100 + "%")
        if class_desc == "circle":
            present_label = "circle"
	    a.write(" with shape identified as circle \n")
        elif class_desc == "square":
            present_label = "square"
	    a.write(" with shape identified as square \n")
        elif class_desc == "star":
            present_label = "star"
	    a.write(" with shape identified as star \n")
        elif class_desc == "triangle":
            present_label = "triangle"
	    a.write(" with shape identified as triangle \n")
        else:
            present_label = past_label           
    else:
        present_label = past_label
        a.write("Low Confidence: " + str(confidence)*100 + "% \n")

    # overlay the result on the image	
    font.OverlayText(img, img.width, img.height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
	
    #record result in txt file
    if present_label != past_label:
        b.write("This is a " + present_label + " with a confidence level of " + str(confidence) + "\n")
        past_label = present_label

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()
        
    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

a.close()
b.close()	


