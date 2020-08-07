#!/bin/bash
# error: object file .git/objects/b8

find .git/objects/ -type f -empty | xargs rm
git fetch -p
git fsck --full
