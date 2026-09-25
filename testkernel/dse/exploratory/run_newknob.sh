#!/bin/bash
cd /raid/sp56429/chia_hackathon/testkernel/dse
/usr/bin/python3.10 newknob_probe.py > newknob_probe.log 2>&1
echo "EXIT=$?" >> newknob_probe.log
