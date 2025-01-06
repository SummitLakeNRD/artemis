# Artemis
**Author**: Keane Flynn
**Organization**: Summit Lake Paiute Tribe
**Date**: 01/06/2025
**Contact**: kmflynn24@berkeley.edu


## Overview
Artemis is real-time IP camera based (will also work with video files) game cam\
program written in python that leverages the YOLOv5 model "Megadetector" to\
summarize wildlife findings. This program is capable of outputting tabular data\
to either individal json files or to a PostgreSQL database. If the former is\
selected, images and files will be output under the same filename prefix. If the\
latter is selected summarized data and images (broken into byte array format)\
will all be appended to the selected Postgres database table.

## Hardware
This software is capable of running on any modern computer. However, like any\ 
other software leveraging neural networks, running on dedicated, CUDA-enabled\
Nvidia GPUs will speed up processing by at least an order of magnitude. SLPT\ 
staff will primarily run this on the Lambda server with 8 GPUs, dual CPUs, and\
tons of RAM so there will be no issues with bottlenecking. Whether the processing\
is performed on your CPU or GPU is dependent upon if PyTorch was installed with\
GPU backend capabilities; the program will dynamically adapt.

## Input & Output
All input parameters can be found using the `python -h` flag 

### Inputs
*Video Source*: Can be either an rtsp url to a video stream or file path to video.
*Model*: File path to Megadetector (or custom) model.
*Confidence Threshold*: 0-1 decimal value for confidence to flag as detection.
*GPU Device*: GPUs have device numbers 0-whatever depending upon how many you have,\
please select accordingly. SLPT tribe will use GPU block 0-3 for inference.
*Site Name*: Name of game camera site.
*Latitude*: Latitude of game camera.
*Longitude*: Longitude of game camera.
*Output Type*: Default is 'postgres' and will append to database, anything else\
entered will output text and image files.
*Output with Bounding Boxes*: Default is 'no' and images will not have boxes\
drawn around detections. 'yes' will draw a box around detections.

### Outputs
*PostgreSQL*: Datetime, site name, class, confidence of inference, bounding box\
coordinates (potentially useful for future training data), latitude, longitude,\
and the image encoded into a byte array.
*Files*: Same tabular data as above written into a json file, but the image is\
written to an individual png file with the same filename prefix as the json.

## How To Use
Issue the following command in your terminal to clone the repository:
```
git clone https://github.com/SummitLakeNRD/Artemis.git
```
*Note: There might be an issue with GitHub large file storage (LFS), if that is\
the case it might be better to simply download the zip file and send the file\ 
to the desired location.*

You will then need to create a python virtual environment to install the correct\
dependencies.
```
python -m venv <VIRTUAL_ENVIRONMENT_NAME>
```
Then activate the virtual environment:
```
source <VIRTUAL_ENVIRONMENT_NAME>/bin/activate
```
Then install the necessary dependencies for this repository:
```
pip install -r requirements.txt
```
While this program will be run from a systemd service file, it can be tested\
by running `python artemis.py -h` to view the necessary positional arguments.\
If you would like to run a test video, running `./testVid` on a linux system\
will perform a sample inference on a video of a coyote from the SLPT reservation.\

### Running From Systemd
This program is primarily designed to be run on a server from the linux systemd.\
To SLPT staff, in short, this means that given the proper instruction, it will\
run continuously on start up and restart if the program crashes for some reason.\
To make this work, you will need to modify the systemd file located in the\
service directory and place it in the following directory:
```
/etc/systemd/system/
```
You will then need to modify the permissions of the systemd service file:
```
sudo chmod 644 /etc/systemd/system/artemis.service
```
Then proceed to reload the daemon, activate the service, and start it:
```
sudo systemctl daemon-reload
sudo systemctl enable artemis.service
sudo systemctl start artemis.service
```
