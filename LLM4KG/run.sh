#!/bin/bash

python3 llm2kg_gen.py > log.out 2> log.err


if [ $? -eq 0 ]; then
    echo "Script executed successfully."
else
    echo "Script encountered an error. Check log.err for details."
fi
