#!/bin/bash

#read -p "Enter commit message: " msg

i += 1
git add .
git commit -m "fixing $i"
git push origin main
