// player.cu - one GPU competitor for Mini Chinese Checkers.
//
//   ./player init                      create a fresh game (state.txt)
//   ./player <1|2> <greedy|lookahead>  play as player 1 or 2
//
// Two player processes run at the same time and take turns through
// state.txt. Every read/write of the shared files happens while holding
// game.lock (created exclusively), so the processes never clash.
// Built with g++ instead of nvcc, the kernels run as plain CPU loops.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
static void sleepMs(int ms) { Sleep(ms); }
#else
#include <unistd.h>
static void sleepMs(int ms) { usleep(ms * 1000); }
#endif
#include "board.h"

#define MAX_PLIES 200
#define STATE "state.txt"
#define LOG   "game_log.txt"
#define LOCK  "game.lock"

// ---------------------------------------------------------------- scoring
// Greedy: score the board right after each of my moves.
HD inline int greedyScore(Board b, int p, Move m) {
    applyMove(b, m);
    return evaluate(b, p);
}

// Lookahead: play my move, then assume the opponent answers with the reply
// that is worst for me. Score = the board after that reply.
HD inline int lookaheadScore(Board b, int p, Move m) {
    applyMove(b, m);
    if (hasWon(b, p)) return 100000;
    Move replies[MAX_MOVES];
    int n = genMoves(b, 3 - p, replies);
    int worst = evaluate(b, p);
    for (int j = 0; j < n; j++) {
        Board b2 = b;
        applyMove(b2, replies[j]);
        int s = evaluate(b2, p);
        if (j == 0 || s < worst) worst = s;
    }
    return worst;
}

#ifdef __CUDACC__
// One GPU thread per candidate move.
__global__ void scoreKernel(Board b, int p, int strategy, const Move *moves, int n, int *scores) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scores[i] = strategy == 0 ? greedyScore(b, p, moves[i]) : lookaheadScore(b, p, moves[i]);
}
#endif

// Scores all moves and returns the time spent in ms.
float scoreMoves(const Board &b, int p, int strategy, const Move *moves, int n, int *scores) {
#ifdef __CUDACC__
    Move *dMoves; int *dScores;
    cudaMalloc(&dMoves, n * sizeof(Move));
    cudaMalloc(&dScores, n * sizeof(int));
    cudaMemcpy(dMoves, moves, n * sizeof(Move), cudaMemcpyHostToDevice);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int threads = 64, blocks = (n + threads - 1) / threads;
    scoreKernel<<<blocks, threads>>>(b, p, strategy, dMoves, n, dScores);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); }
    cudaMemcpy(scores, dScores, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dMoves); cudaFree(dScores);
    return ms;
#else
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
        scores[i] = strategy == 0 ? greedyScore(b, p, moves[i]) : lookaheadScore(b, p, moves[i]);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
#endif
}

// ---------------------------------------------------------- shared files
struct State { int turn, ply, status; Board b; };  // status: 0 playing, 1/2 winner, 3 draw

// O_CREAT|O_EXCL fails if the file exists, so only one process gets the lock.
void lock()   { int fd; while ((fd = open(LOCK, O_CREAT | O_EXCL | O_WRONLY, 0644)) < 0) sleepMs(20); close(fd); }
void unlock() { remove(LOCK); }

bool readState(State &s) {
    FILE *f = fopen(STATE, "r");
    if (!f) return false;
    char cells[CELLS + 1];
    bool ok = fscanf(f, "%d %d %d %49s", &s.turn, &s.ply, &s.status, cells) == 4;
    fclose(f);
    for (int i = 0; ok && i < CELLS; i++) s.b.cell[i] = cells[i] - '0';
    return ok;
}

void writeState(const State &s) {
    FILE *f = fopen(STATE, "w");
    fprintf(f, "%d %d %d ", s.turn, s.ply, s.status);
    for (int i = 0; i < CELLS; i++) fputc('0' + s.b.cell[i], f);
    fputc('\n', f);
    fclose(f);
}

void printBoard(const Board &b) {
    const char sym[3] = {'.', 'A', 'B'};
    for (int r = 0; r < N; r++) {
        printf("   %*s", r, "");
        for (int c = 0; c < N; c++) printf("%c ", sym[(int)b.cell[r * N + c]]);
        printf("\n");
    }
}

int pegsHome(const Board &b, int p) {
    int n = 0;
    for (int i = 0; i < CELLS; i++) n += inGoal(p, i) && b.cell[i] == p;
    return n;
}

// ------------------------------------------------------------------ main
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "init")) {
        State s = {1, 0, 0, {}};
        initBoard(s.b);
        remove(LOCK);
        writeState(s);
        FILE *log = fopen(LOG, "w");
        fprintf(log, "START 0 0 0 0 ");
        for (int i = 0; i < CELLS; i++) fputc('0' + s.b.cell[i], log);
        fputc('\n', log); fclose(log);
        printf("New game created.\n");
        printBoard(s.b);
        return 0;
    }
    if (argc != 3) { printf("usage: %s init | %s <1|2> <greedy|lookahead>\n", argv[0], argv[0]); return 1; }

    int me = atoi(argv[1]);
    int strategy = strcmp(argv[2], "greedy") == 0 ? 0 : 1;
    const char *name = me == 1 ? "A" : "B";
    Move *moves = new Move[MAX_MOVES];
    int *scores = new int[MAX_MOVES];

    while (true) {
        lock();
        State s;
        if (!readState(s)) { unlock(); fprintf(stderr, "No game. Run: player init\n"); return 1; }
        if (s.status != 0) {
            unlock();
            if (s.status == 3) printf("[%s] Game over: draw.\n", name);
            else printf("[%s] Game over: player %s wins!\n", name, s.status == 1 ? "A" : "B");
            break;
        }
        if (s.turn != me) { unlock(); sleepMs(50); continue; }

        // --- my turn: list moves, score them on the GPU, pick the best
        int n = genMoves(s.b, me, moves);
        int best = -1;
        float ms = 0;
        if (n > 0) {
            ms = scoreMoves(s.b, me, strategy, moves, n, scores);
            best = 0;
            for (int i = 1; i < n; i++) if (scores[i] > scores[best]) best = i;
            applyMove(s.b, moves[best]);
        }
        s.ply++;
        s.turn = 3 - me;
        if (hasWon(s.b, me)) s.status = me;
        else if (s.ply >= MAX_PLIES) {   // move cap: most pegs home wins
            int a = pegsHome(s.b, 1), bb = pegsHome(s.b, 2);
            s.status = a > bb ? 1 : bb > a ? 2 : 3;
        }

        Move m = best >= 0 ? moves[best] : Move{-1, -1};
        FILE *log = fopen(LOG, "a");
        fprintf(log, "MOVE %d %d %d %d ", s.ply, me, m.from, m.to);
        for (int i = 0; i < CELLS; i++) fputc('0' + s.b.cell[i], log);
        fputc('\n', log); fclose(log);

        printf("\nMove %d - player %s (%s): (%d,%d) -> (%d,%d)  score %d  [%d moves scored in %.3f ms]\n",
               s.ply, name, argv[2], m.from / N, m.from % N, m.to / N, m.to % N,
               best >= 0 ? scores[best] : 0, n, ms);
        printBoard(s.b);
        fflush(stdout);
        writeState(s);
        unlock();
    }
    delete[] moves; delete[] scores;
    return 0;
}
