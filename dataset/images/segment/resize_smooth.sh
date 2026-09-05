#!/bin/bash

# run in cr_bot

# ./KataCR/Clash-Royale-Detection-Dataset/images/segment/resize_smooth.sh

python KataCR/Clash-Royale-Detection-Dataset/images/segment/resize.py
python KataCR/Clash-Royale-Detection-Dataset/images/segment/make_smooth.py

cp -r contour_fade_contour_10000px KataCR/Clash-Royale-Detection-Dataset/images/segment
rm -r contour_fade_contour_10000px