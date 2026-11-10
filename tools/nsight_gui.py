"""Drive and capture an X11 window under WSLg.
usage: xui.py <window-name-substring> <cmd> [args...] ...
cmds: shot PATH | click X Y | dclick X Y | rclick X Y | scroll X Y N(+down/-up) | key KEYSYM[+MOD]
      | type TEXT | resize W H | sleep S | raise
Coordinates are relative to the window's top-left.
"""
import sys, time
from Xlib import X, XK, display
from Xlib.ext import xtest
from PIL import Image

d = display.Display()
root = d.screen().root


def find_all(win, name, out):
    try:
        n = win.get_wm_name()
        geo = win.get_geometry()
        if n and name in n and win.get_attributes().map_state == X.IsViewable:
            out.append((geo.width * geo.height, win))
    except Exception:
        pass
    for c in win.query_tree().children:
        find_all(c, name, out)
    return out


cands = sorted(find_all(root, sys.argv[1], []), key=lambda t: -t[0])
w = cands[0][1] if cands else None
if not w:
    sys.exit("window not found")


def origin():
    t = w.translate_coords(root, 0, 0)
    return -t.x, -t.y


def move(x, y):
    ox, oy = origin()
    xtest.fake_input(d, X.MotionNotify, x=ox + x, y=oy + y)
    d.sync()
    time.sleep(0.05)


def button(b, n=1):
    for _ in range(n):
        xtest.fake_input(d, X.ButtonPress, b)
        xtest.fake_input(d, X.ButtonRelease, b)
        d.sync()
        time.sleep(0.05)


def key(spec):
    parts = spec.split("+")
    codes = [d.keysym_to_keycode(XK.string_to_keysym(p)) for p in parts]
    for c in codes:
        xtest.fake_input(d, X.KeyPress, c)
    for c in reversed(codes):
        xtest.fake_input(d, X.KeyRelease, c)
    d.sync()
    time.sleep(0.05)


args = sys.argv[2:]
i = 0
while i < len(args):
    c = args[i]
    if c == "shot":
        g = w.get_geometry()
        raw = w.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xFFFFFFFF)
        Image.frombytes("RGB", (g.width, g.height), raw.data, "raw", "BGRX").save(args[i + 1])
        print("saved", args[i + 1], g.width, g.height)
        i += 2
    elif c == "rootshot":  # whole X screen, cropped to the window area + margin (shows popups)
        g = root.get_geometry()
        raw = root.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xFFFFFFFF)
        im = Image.frombytes("RGB", (g.width, g.height), raw.data, "raw", "BGRX")
        ox, oy = origin()
        wg = w.get_geometry()
        im.crop((max(ox - 50, 0), max(oy - 50, 0), min(ox + wg.width + 50, g.width), min(oy + wg.height + 50, g.height))).save(args[i + 1])
        print("saved root crop", args[i + 1], "origin", ox, oy)
        i += 2
    elif c in ("click", "dclick", "rclick"):
        move(int(args[i + 1]), int(args[i + 2]))
        button(3 if c == "rclick" else 1, 2 if c == "dclick" else 1)
        i += 3
    elif c == "drag":
        x0, y0, x1, y1 = (int(v) for v in args[i + 1:i + 5])
        move(x0, y0)
        xtest.fake_input(d, X.ButtonPress, 1)
        d.sync()
        for s in range(1, 11):
            move(x0 + (x1 - x0) * s // 10, y0 + (y1 - y0) * s // 10)
        xtest.fake_input(d, X.ButtonRelease, 1)
        d.sync()
        i += 5
    elif c == "scroll":
        move(int(args[i + 1]), int(args[i + 2]))
        n = int(args[i + 3])
        button(5 if n > 0 else 4, abs(n))
        i += 4
    elif c == "key":
        key(args[i + 1])
        i += 2
    elif c == "type":
        for ch in args[i + 1]:
            key(ch if ch.isalnum() else {"_": "underscore", ".": "period", "/": "slash", "-": "minus", " ": "space"}.get(ch, ch))
        i += 2
    elif c == "resize":
        w.configure(width=int(args[i + 1]), height=int(args[i + 2]))
        d.sync()
        i += 3
    elif c == "raise":
        w.configure(stack_mode=X.Above)
        w.set_input_focus(X.RevertToParent, X.CurrentTime)
        d.sync()
        i += 1
    elif c == "sleep":
        time.sleep(float(args[i + 1]))
        i += 2
    else:
        sys.exit("bad cmd " + c)
