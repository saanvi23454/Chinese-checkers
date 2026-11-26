#!/bin/bash
# Start a new game and run both GPU players at the same time.
# They take turns through state.txt, guarded by game.lock.
set -e
make -s
./player init
./player 1 greedy &
./player 2 lookahead &
wait
python3 visualize.py && echo "Open game.gif to watch the replay."
