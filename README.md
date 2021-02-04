## Medit-Seminar
This repository contains the code for our seminar: "Light Control with Real-time Music Detection".

Inspiration for beat detection: https://github.com/naztronaut/dancyPi-audio-reactive-led

Inspiration for adaptive threshold: https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data

# Setup
Before being able to send MIDI messages you need to do the following:
navigate to /boot/cmdline.txt and delete the following: console=serial0,115200

in /boot/config.txt add the following (i.e. at the end of the file):
enable_uart=1
dtoverlay=pi3-miniuart-bt
dtoverlay=midi-uart0


Before starting: execute install.py with sudo python3 install.py

The main file is lightcontrol.py: sudo python3 lightcontrol.py

