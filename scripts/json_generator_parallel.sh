#!/bin/bash

SCRIPT="python json_for_RDM_generator_vFF_server_ver.py"

for i in {1..27}; do
    gnome-terminal -- bash -c "$SCRIPT $i; exec bash" &
done
