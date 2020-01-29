#!/bin/bash
set -e
echo "Setup GPU memory manager survey"

echo "Clone Halloc"
git clone https://github.com/canonizer/halloc.git frameworks/halloc/repository