# Mini Chinese Checkers: GPU vs GPU

Two CUDA programs play a smaller version of Chinese Checkers against each other.

## The game
- 7×7 diamond board on a hex grid (49 holes, up to 6 neighbours each).
- Each player has 6 pegs in opposite corners. Player **A** starts top-left, player **B** starts bottom-right.
- On your turn, move one peg one step to an empty neighbouring hole, or jump over any peg into the empty hole right behind it. Jumps can be chained.
- The first player to fill the opposite corner wins. After 200 moves the game stops and whoever has more pegs home wins.

## The two GPU players
Each turn, a player lists its legal moves (all steps and jump chains, found with a BFS) and sends them to the GPU. Each **CUDA thread scores one candidate move**, and the player picks the move with the highest score.

| Player | Strategy | What each thread does |
|---|---|---|
| A | `greedy` | Applies the move and scores the resulting board |
| B | `lookahead` | Applies the move, tries every opponent reply, and keeps the worst result (min-max, depth 2) |

The board score is `10 × (opponent distance − my distance) − 3 × (distance of my furthest-behind peg)`, where distance is the hex distance to the goal corner. A win scores ±100000.

## Taking turns
Both players run as separate processes at the same time:
- `state.txt` holds whose turn it is, the move number, game status and the board.
- `game.lock` is created with `O_CREAT|O_EXCL`, so only one process can read or write the shared files at a time.
- A player waits until `state.txt` says it's their turn. Then it makes its move, appends the move to `game_log.txt` and hands the turn over.

Because each player only uses the GPU on its own turn, both can share a single GPU, such as the one in the Coursera lab.

## Files
| File | Purpose |
|---|---|
| `board.h` | Board, move generation, evaluation (`__host__ __device__`) |
| `player.cu` | Scoring kernel, turn-taking, terminal output |
| `run_game.sh` | Builds the code, starts a game and runs both players |
| `visualize.py` | Turns `game_log.txt` into `game.gif` |

## Run
```bash
bash run_game.sh        # needs nvcc; output is printed live in the terminal
```
To run without a GPU, use `make cpu`. The same code compiles with g++ and the kernel runs as a loop on the CPU.

## Output
After every move the terminal shows the move, its score, how many moves the GPU scored and how long the kernel took, then the board:
```
Move 1 - player A (greedy): (0,1) -> (0,3)  score -16  [10 moves scored in 0.050 ms]
   A . A A . . .
    A A . . . . .
     A . . . . . .
      . . . . . . .
       . . . . . . B
        . . . . . B B
         . . . . B B B
```
`visualize.py` turns the game log into `game.gif`, an animated replay. Red pegs are A and blue pegs are B, and the two holes involved in each move are outlined.
