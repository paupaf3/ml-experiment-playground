#!/bin/bash

# Check if filename was provided
if [ -z "$1" ]; then
	echo "Usage: ./run.sh <filename.c>"
	exit 1
fi

FILE=$1
NAME=${FILE%.c} # This strips the ".c" off the end

echo "--- Compiling $FILE... ---"
gcc "$FILE" -o "$NAME" -lm

if [ $? -eq 0 ]; then
	echo "--- Success! Running $NAME... ---"
	./"$NAME"
else
	echo "--- Compilation Failed! ---"
fi
