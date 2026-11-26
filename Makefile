player: player.cu board.h
	nvcc -O2 -o player player.cu

# Same code without a GPU (kernels run as CPU loops), for testing.
cpu: player.cu board.h
	g++ -x c++ -O2 -o player player.cu

clean:
	rm -f player player.exe state.txt game.lock game_log.txt player_*.txt game.gif
