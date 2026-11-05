"""Tkinter desktop app: upload an exam video, track students, flag cheating and raise alerts.

Run:  python desktop_app.py
Headless smoke test:  xvfb-run -a python desktop_app.py --autorun video.mp4 --snapshot shot.png
"""
import argparse
import os
import platform
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.database import AlertDatabase
from src.detector import CheatingMonitor, DetectorConfig, format_notification

APP_TITLE = "Cheating Tracking and Alert System"
ROOT = Path(__file__).resolve().parent


class CheatingApp(tk.Tk):
    def __init__(self, autorun: str | None = None, snapshot: str | None = None):
        super().__init__()
        self.snapshot = snapshot   # optional: save a screenshot of the window when processing ends (smoke test)
        self.title(APP_TITLE)
        self.geometry("1260x720")
        self.minsize(1000, 600)
        self._set_icon()

        self.cfg = DetectorConfig(
            weights=str(ROOT / "models" / "best.pt"),
            screenshot_dir=str(ROOT / "alerts" / "screenshots"),
            db_path=str(ROOT / "alerts" / "alerts.db"),
        )
        self.db = AlertDatabase(self.cfg.db_path)
        self.messages: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.monitor = None
        self.worker = None
        self._photo = None
        self._toasts: list = []
        self.conf_var = tk.DoubleVar(value=self.cfg.conf)
        self.status_var = tk.StringVar(value="Ready. Click 'Upload Video' to start.")

        self._build_ui()
        self.refresh_history()
        self.after(30, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if autorun:
            self.after(500, lambda: self._start(autorun))

    # ------------------------------------------------------------------ UI
    def _set_icon(self):
        icon = ROOT / "assets" / "app_icon.png"
        try:
            if icon.exists():
                self.iconphoto(True, tk.PhotoImage(file=str(icon)))
        except Exception:
            pass

    def _build_ui(self):
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")
        self.btn_upload = ttk.Button(bar, text="Upload Video", command=self.upload_video)
        self.btn_upload.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="Open Screenshots Folder", command=self.open_folder).pack(side="left")
        ttk.Label(bar, text="  Confidence:").pack(side="left")
        ttk.Scale(bar, from_=0.1, to=0.9, variable=self.conf_var, length=160).pack(side="left")
        ttk.Label(bar, textvariable=self.conf_var, width=5).pack(side="left")

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        self.canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        body.add(left, weight=3)

        right = ttk.Frame(body)
        ttk.Label(right, text="Alert History", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        cols = ("time", "behavior", "confidence", "screenshot")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        for col, title, width in (("time", "Date / Time", 140), ("behavior", "Behavior", 110),
                                  ("confidence", "Conf.", 55), ("screenshot", "Screenshot", 200)):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda _e: self.view_screenshot())
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="View Screenshot", command=self.view_screenshot).pack(side="left")
        ttk.Button(buttons, text="Refresh", command=self.refresh_history).pack(side="left", padx=6)
        body.add(right, weight=2)

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=8)
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=6).pack(fill="x")

    # ------------------------------------------------------------- actions
    def upload_video(self):
        path = filedialog.askopenfilename(
            title="Select exam video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self._start(path)

    def _start(self, path: str):
        if not Path(self.cfg.weights).exists():
            messagebox.showerror(APP_TITLE, f"Model weights not found:\n{self.cfg.weights}")
            return
        self.stop_event.clear()
        self.btn_upload.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_var.set(f"Processing {Path(path).name} ...")
        self.worker = threading.Thread(target=self._run_video, args=(path,), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Stopping ...")

    def _run_video(self, path: str):
        try:
            if self.monitor is None:
                self.messages.put(("status", "Loading model ..."))
                self.monitor = CheatingMonitor(
                    self.cfg, db=self.db, on_alert=lambda e: self.messages.put(("alert", e)))
            self.monitor.cfg.conf = round(self.conf_var.get(), 2)
            for item in self.monitor.stream_video(path, stop_event=self.stop_event):
                total = item["total_frames"] or 1
                self.messages.put(("frame", item["frame"], item["frame_index"] / total))
            self.messages.put(("done", Path(path).name))
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def _poll(self):
        last_frame, last_progress = None, 0.0
        try:
            while True:
                msg = self.messages.get_nowait()
                kind = msg[0]
                if kind == "frame":
                    last_frame, last_progress = msg[1], msg[2]
                elif kind == "alert":
                    self.refresh_history()
                    self._toast(msg[1])
                elif kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "done":
                    self._finish(f"Finished {msg[1]}. Alerts are saved in the history.")
                    if self.snapshot:
                        self.after(1000, self._snapshot_and_quit)
                elif kind == "error":
                    self._finish("Error")
                    if self.snapshot:
                        print("Processing error:", msg[1])
                        self._on_close()
                        return
                    messagebox.showerror(APP_TITLE, msg[1])
        except queue.Empty:
            pass
        if last_frame is not None:
            self._show_frame(last_frame)
            self.progress["value"] = min(last_progress, 1.0) * 100
        self.after(30, self._poll)

    def _finish(self, text: str):
        self.btn_upload.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_var.set(text)

    def _show_frame(self, frame):
        h, w = frame.shape[:2]
        cw, ch = max(self.canvas.winfo_width(), 320), max(self.canvas.winfo_height(), 240)
        scale = min(cw / w, ch / h)
        resized = cv2.resize(frame, (max(int(w * scale), 1), max(int(h * scale), 1)))
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

    def _toast(self, event):
        """Non-blocking desktop notification shown at the bottom-right of the window."""
        top = tk.Toplevel(self)
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        box = tk.Frame(top, bg="#b71c1c", padx=16, pady=12)
        box.pack()
        tk.Label(box, text=format_notification(event), justify="left", fg="white", bg="#b71c1c",
                 font=("Segoe UI", 10, "bold")).pack()
        self.update_idletasks()
        offset = sum(t.winfo_reqheight() + 8 for t in self._toasts if t.winfo_exists())
        x = self.winfo_rootx() + self.winfo_width() - top.winfo_reqwidth() - 20
        y = self.winfo_rooty() + self.winfo_height() - top.winfo_reqheight() - 60 - offset
        top.geometry(f"+{x}+{max(y, 0)}")
        self._toasts.append(top)
        self.bell()

        def close():
            if top.winfo_exists():
                top.destroy()
            self._toasts[:] = [t for t in self._toasts if t.winfo_exists()]

        top.after(6000, close)

    # ------------------------------------------------------------- history
    def refresh_history(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in self.db.fetch_all(limit=500):
            self.tree.insert("", "end", iid=str(r["id"]), values=(
                r["detected_at"].replace("T", " "), r["behavior"],
                f"{r['confidence']:.2f}", r["screenshot_path"]))

    def view_screenshot(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo(APP_TITLE, "Select an alert first.")
            return
        path = Path(str(self.tree.item(selected[0])["values"][3]))
        if not path.exists():
            messagebox.showwarning(APP_TITLE, f"File not found:\n{path}")
            return
        win = tk.Toplevel(self)
        win.title(path.name)
        img = Image.open(path)
        img.thumbnail((1000, 700))
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(win, image=photo)
        label.image = photo
        label.pack()

    def open_folder(self):
        folder = Path(self.cfg.screenshot_dir)
        folder.mkdir(parents=True, exist_ok=True)
        system = platform.system()
        if system == "Windows":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", str(folder)])
        else:
            subprocess.run(["xdg-open", str(folder)])

    def _snapshot_and_quit(self):
        """Used by the headless smoke test: save a picture of the window and exit."""
        try:
            from PIL import ImageGrab

            self.update()
            x, y = self.winfo_rootx(), self.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + self.winfo_width(), y + self.winfo_height())).save(self.snapshot)
        except Exception as exc:
            print("Snapshot failed:", exc)
        self._on_close()

    def _on_close(self):
        self.stop_event.set()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--autorun", help="Process this video right after start-up (smoke test)")
    parser.add_argument("--snapshot", help="Save a window screenshot here when processing ends, then exit")
    args = parser.parse_args()
    CheatingApp(autorun=args.autorun, snapshot=args.snapshot).mainloop()


if __name__ == "__main__":
    main()
