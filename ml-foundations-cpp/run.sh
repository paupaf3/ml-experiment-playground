#!/bin/bash

# Check if a filename was provided
if [ -z "$1" ]; then
	echo "Usage: ./run.sh <filename.cpp>"
	exit 1
fi

FILE=$1
NAME=${FILE%.*}

echo "--- Compiling $FILE ---"
# -Wall enables "all warnings"
# -lm links to math library
g++ "$FILE" -o "$NAME" -Wall -lm

if [ $? -eq 0 ]; then
	echo "--- Success! Running $NAME... ---"
	./"$NAME"
else
	echo "--- Compilation Failed! ---"
fi
