#!/bin/bash
cd /raid/sp56429/chia_hackathon/testkernel/dse
/usr/bin/python3.10 linewave_probe.py > linewave_probe.log 2>&1
echo "EXIT=$?" >> linewave_probe.log
