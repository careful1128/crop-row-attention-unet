#!/bin/bash 
 ps -ef | grep ddp 
 ps -ef | grep ddp | awk '{print $2}' | xargs kill -9 