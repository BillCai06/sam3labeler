# Labeler — Local Network Setup

Run the labeler on one GPU machine and access it from any browser on the same network.

---

## 1. Start the server

```bash
cd /home/bill/qwen3vl2sam
python run_batch.py --labeler --port 7777
```

The server binds to `0.0.0.0` (all interfaces) by default, so it is immediately reachable from the local network — no extra config needed.

---

## 2. Find the host machine's IP

On the server machine:

```bash
ip route get 1 | awk '{print $7; exit}'
# example output: 192.168.50.173
```

Or check `ip addr` / `ifconfig` and look for your LAN interface (usually `eth0` or `enp*`).

---

## 3. Open from another machine

On any machine on the same network, open a browser and go to:

```
http://192.168.50.173:7777
```

Replace `192.168.50.173` with whatever IP the step above returned.

---

## 4. Firewall — open the port (if needed)

If other machines can't connect, the firewall is likely blocking port 7777.

**Ubuntu / Debian (ufw):**
```bash
sudo ufw allow 7777/tcp
sudo ufw reload
```

**Check it worked:**
```bash
# From another machine
curl -s http://192.168.50.173:7777/api/datasets | head -c 100
```

---

## 5. Keep it running (optional)

To keep the server alive after closing the terminal:

**tmux (simplest):**
```bash
tmux new -s labeler
python run_batch.py --labeler --port 7777
# Detach: Ctrl+B, then D
# Re-attach later: tmux attach -t labeler
```

**systemd service (persistent across reboots):**

Create `/etc/systemd/system/labeler.service`:
```ini
[Unit]
Description=SAM3 Labeler
After=network.target

[Service]
User=bill
WorkingDirectory=/home/bill/qwen3vl2sam
ExecStart=/usr/bin/python3 run_batch.py --labeler --port 7777
Restart=on-failure
Environment=LABELER_CONFIG=/home/bill/qwen3vl2sam/config.yaml

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now labeler
sudo systemctl status labeler
```

---

## 6. Multi-user notes

| Feature | Concurrent users |
|---------|-----------------|
| Browse images, edit annotations, save | Unlimited |
| SAM inference (`/api/sam`, point, propagate) | **1 at a time** — requests queue automatically |

Multiple people editing the **same image** at the same time will overwrite each other's saves (last write wins). Coordinate by working on different images or different datasets.

**Recommended workflow for a team:**
- Split the dataset into per-person subfolders, or
- Work sequentially on the same folder (one person at a time per image)

---

## 7. Shared dataset path

The dataset folder (e.g. `outputs/run_20260318/`) must be accessible on the server machine. If annotators are on different machines, mount it via NFS or SMB:

```bash
# Server: share the outputs folder
sudo apt install nfs-kernel-server
echo "/home/bill/qwen3vl2sam/outputs  192.168.50.0/24(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -ra

# Client: mount it
sudo mount 192.168.50.173:/home/bill/qwen3vl2sam/outputs /mnt/labeler_outputs
```

Then point the labeler at `/mnt/labeler_outputs` from the client — though it's simpler to just run the browser on the server machine and let everyone use the web UI.
