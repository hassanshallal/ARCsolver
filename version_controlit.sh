#!/bin/bash

git add *.py
git add *.ipynb
git add ./assets
git commit -m "$1"
git remote -v
git push -u origin master

