// board.h - Mini Chinese Checkers rules, shared by host and GPU code.
//
// Board: 7x7 diamond on a hex grid (axial coords r,c). Each hole has up to
// 6 neighbours. Player 1 starts in the corner near (0,0) and must reach the
// corner near (6,6); Player 2 does the opposite. 6 pegs each.
#pragma once
#include <string.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#define N        7
#define CELLS    (N * N)
#define PEGS     6
#define MAX_MOVES (PEGS * CELLS)   // upper bound: every peg to every hole

struct Board { char cell[CELLS]; };  // 0 = empty, 1 = player 1, 2 = player 2
struct Move  { int from, to; };

HD inline bool onBoard(int r, int c) { return r >= 0 && r < N && c >= 0 && c < N; }

// The 6 hex directions in axial coordinates.
HD inline void dir(int d, int &dr, int &dc) {
    const int R[6] = {0, 0, 1, -1, 1, -1};
    const int C[6] = {1, -1, 0, 0, -1, 1};
    dr = R[d]; dc = C[d];
}

// Starting triangle of a player (also the other player's goal).
HD inline bool inHome(int p, int idx) {
    int r = idx / N, c = idx % N;
    return p == 1 ? (r + c <= 2) : (r + c >= 2 * (N - 1) - 2);
}
HD inline bool inGoal(int p, int idx) { return inHome(3 - p, idx); }

// Hex distance from a hole to the far tip of player p's goal corner.
HD inline int distToGoal(int p, int idx) {
    int r = idx / N, c = idx % N;
    return p == 1 ? (2 * (N - 1) - r - c) : (r + c);
}

HD inline void initBoard(Board &b) {
    for (int i = 0; i < CELLS; i++)
        b.cell[i] = inHome(1, i) ? 1 : inHome(2, i) ? 2 : 0;
}

HD inline bool hasWon(const Board &b, int p) {
    for (int i = 0; i < CELLS; i++)
        if (inGoal(p, i) && b.cell[i] != p) return false;
    return true;
}

// All holes reachable by the peg at `from`: one step, or a chain of jumps.
// Writes 1 into reach[to]. Uses a small BFS stack (no recursion -> GPU safe).
HD inline void reachable(const Board &b, int from, char reach[CELLS]) {
    for (int i = 0; i < CELLS; i++) reach[i] = 0;
    int r0 = from / N, c0 = from % N, dr, dc;
    for (int d = 0; d < 6; d++) {                // single steps
        dir(d, dr, dc);
        int r = r0 + dr, c = c0 + dc;
        if (onBoard(r, c) && b.cell[r * N + c] == 0) reach[r * N + c] = 1;
    }
    char seen[CELLS] = {0};
    int stack[CELLS], top = 0;
    stack[top++] = from; seen[from] = 1;
    while (top > 0) {                            // jump chains
        int cur = stack[--top], r = cur / N, c = cur % N;
        for (int d = 0; d < 6; d++) {
            dir(d, dr, dc);
            int mr = r + dr, mc = c + dc, jr = r + 2 * dr, jc = c + 2 * dc;
            if (!onBoard(jr, jc)) continue;
            int mid = mr * N + mc, dst = jr * N + jc;
            if (b.cell[mid] != 0 && mid != from && (b.cell[dst] == 0 || dst == from) && !seen[dst]) {
                seen[dst] = 1; reach[dst] = 1; stack[top++] = dst;
            }
        }
    }
    reach[from] = 0;
}

// List every legal move of player p. Returns the count.
HD inline int genMoves(const Board &b, int p, Move *out) {
    int n = 0; char reach[CELLS];
    for (int f = 0; f < CELLS; f++) {
        if (b.cell[f] != p) continue;
        reachable(b, f, reach);
        for (int t = 0; t < CELLS; t++)
            if (reach[t]) { out[n].from = f; out[n].to = t; n++; }
    }
    return n;
}

HD inline void applyMove(Board &b, Move m) { b.cell[m.to] = b.cell[m.from]; b.cell[m.from] = 0; }

// Board score from player p's view (higher is better):
// opponent's remaining distance minus mine, with an extra penalty for the
// peg that is furthest behind so no peg gets left stranded.
HD inline int evaluate(const Board &b, int p) {
    if (hasWon(b, p)) return 100000;
    if (hasWon(b, 3 - p)) return -100000;
    int mine = 0, theirs = 0, worst = 0;
    for (int i = 0; i < CELLS; i++) {
        if (b.cell[i] == p)          { int d = distToGoal(p, i); mine += d; if (d > worst) worst = d; }
        else if (b.cell[i] == 3 - p) theirs += distToGoal(3 - p, i);
    }
    return 10 * (theirs - mine) - 3 * worst;
}
