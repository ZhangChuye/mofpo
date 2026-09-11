"""Read logs.json.txt lines from stdin; print one line per end-of-epoch record at a given epoch stride."""
import sys, json, time
stride = int(sys.argv[1]) if len(sys.argv) > 1 else 20
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
t0 = time.time(); last_ep, last_t = None, None
for line in sys.stdin:
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "val_loss" not in d:
        continue
    ep = int(d.get("epoch", -1))
    rate = ""
    if last_ep is not None and ep > last_ep:
        rate = f" ({(time.time() - last_t) / (ep - last_ep) / 60:.1f} min/epoch, ETA {(499 - ep) * (time.time() - last_t) / (ep - last_ep) / 3600:.1f} h)"
    if ep % stride == 0:
        print(f"{tag} epoch {ep} done: train_loss={d.get('train_loss', float('nan')):.4f} val_loss={d['val_loss']:.4f}{rate}", flush=True)
        last_ep, last_t = ep, time.time()
