"""Turn game_log.txt into game.gif (one frame per move). Needs Pillow."""
import sys
from PIL import Image, ImageDraw

N, R, GAP = 7, 16, 40          # board size, hole radius, hole spacing
COLORS = {"0": "#d9d9d9", "1": "#d62728", "2": "#1f77b4"}
NAMES = {"1": "A (greedy)", "2": "B (lookahead)"}

def pos(r, c):
    # Shift each row half a hole so the grid looks hexagonal.
    return 60 + c * GAP + r * GAP / 2, 70 + r * GAP * 0.87

def frame(line):
    kind, ply, player, frm, to, cells = line.split()
    img = Image.new("RGB", (480, 360), "white")
    d = ImageDraw.Draw(img)
    title = "Start" if kind == "START" else f"Move {ply}: player {NAMES[player]}"
    d.text((20, 15), title, fill="black")
    for i, v in enumerate(cells):
        x, y = pos(i // N, i % N)
        ring = "black" if str(i) in (frm, to) and kind == "MOVE" else "#999999"
        d.ellipse([x - R, y - R, x + R, y + R], fill=COLORS[v], outline=ring, width=3 if ring == "black" else 1)
    return img

def main(path="game_log.txt", out="game.gif"):
    frames = [frame(l) for l in open(path) if l.strip()]
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 5, duration=400, loop=0)
    print(f"Saved {out} ({len(frames)} frames)")

if __name__ == "__main__":
    main(*sys.argv[1:])
