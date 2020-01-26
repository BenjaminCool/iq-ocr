#!/bin/bash

read -p 'Image Name: ' image_name
read -p 'Image Label: ' label
docker build --tag ${image_name}:${label} .

