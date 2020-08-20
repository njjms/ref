#!/bin/bash

x=1
y=$((x-1))

while [ $x -ge 0 ]
do
	sleep 1
	echo $x
	x=$((x+1))
done

