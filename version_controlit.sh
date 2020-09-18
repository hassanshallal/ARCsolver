#!/bin/bash

git add *.py
git add *.ipynb
git add *.sh
git add ./assets
git add ./depreciated
git commit -m "$1"
git remote -v
git push -u origin master

