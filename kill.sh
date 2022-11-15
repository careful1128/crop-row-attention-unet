#!/bin/bash 
 ps -ef | grep train_with 
 ps -ef | grep train_with | awk '{print $2}' | xargs kill -9 