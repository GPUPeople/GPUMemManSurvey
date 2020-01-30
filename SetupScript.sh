#!/bin/bash
set -e
echo "Setup GPU memory manager survey"

echo "------------------------"
echo "Setup Halloc"
echo "------------------------"
hallocDir="frameworks/halloc/repository"
if [ ! -d "$hallocDir" ]; then
  # Take action if $DIR exists. #
  echo "Cloning Halloc to ${hallocDir}..."
  git clone https://github.com/canonizer/halloc.git $hallocDir
else
  echo "Halloc already cloned to ${hallocDir}"
fi
