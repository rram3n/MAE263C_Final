#!/usr/bin/env python3
"""
Halftoning robot - GUI
======================

A Tkinter front-end for halftone_robot.py.  Load a photo, set the X/Y/Z
ranges and the halftone knobs (optionally discretizing Z into levels), watch a
live to-scale preview, then export the per-pen CSV toolpaths.

Run:
    python halftone_gui.py

Dependencies: numpy, pillow, matplotlib (tkinter ships with Python).
"""

from __future__ import annotations

import os
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import halftone_robot as hr


class HalftoneGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Halftoning Robot - Dot Toolpath Generator")
        root.geometry("1180x780")

        self.image_path = tk.StringVar(value="")
        self._last_result = None        # (cfg, dots, layout, max_dot)

        # ---- tk variables for every knob ----
        self.v = dict(
            x_min=tk.DoubleVar(value=0.0),   x_max=tk.DoubleVar(value=200.0),
            y_min=tk.DoubleVar(value=0.0),   y_max=tk.DoubleVar(value=250.0),
            z_min=tk.DoubleVar(value=0.0),   z_max=tk.DoubleVar(value=4.0),
            mode=tk.StringVar(value="dark"),
            dot_pitch=tk.DoubleVar(value=3.0),
            cutoff=tk.DoubleVar(value=0.90),
            gamma=tk.DoubleVar(value=1.0),
            max_dot=tk.DoubleVar(value=0.0),
            min_dot=tk.DoubleVar(value=0.0),
            min_coverage=tk.DoubleVar(value=0.04),
            invert=tk.BooleanVar(value=False),
            # adaptive density + budget
            adaptive=tk.BooleanVar(value=False),
            fine_pitch=tk.DoubleVar(value=1.5),
            detail_threshold=tk.DoubleVar(value=0.12),
            budget_on=tk.BooleanVar(value=False),
            max_dots=tk.IntVar(value=3000),
            # Z discretization
            z_levels_on=tk.BooleanVar(value=False),
            z_level_min=tk.IntVar(value=0),
            z_levels=tk.IntVar(value=4),
            # output
            out_prefix=tk.StringVar(value="halftone"),
            auto_update=tk.BooleanVar(value=True),
        )

        self._build_layout()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        # Scrollable control column on the left.
        left = self._make_scrollable(main, width=300)
        self._build_controls(left)

        # Preview on the right.
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(6, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Load an image to begin")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.status = ttk.Label(self.root, text="Ready", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def _make_scrollable(self, parent, width):
        """Create a vertically scrollable column and return the inner frame to
        place widgets into."""
        outer = ttk.Frame(parent)
        outer.pack(side="left", fill="y")

        canvas = tk.Canvas(outer, width=width, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="y", expand=True)
        vbar.pack(side="right", fill="y")

        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Keep the scrollregion in sync with the inner frame, and make the
        # inner frame track the canvas width.
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(window, width=e.width))

        # Mouse-wheel scrolling while the pointer is over the column.
        def _on_wheel(event):
            delta = -1 if (event.num == 5 or event.delta < 0) else 1
            canvas.yview_scroll(-delta, "units")

        def _bind_wheel(_):
            canvas.bind_all("<MouseWheel>", _on_wheel)      # Windows / macOS
            canvas.bind_all("<Button-4>", _on_wheel)        # Linux up
            canvas.bind_all("<Button-5>", _on_wheel)        # Linux down

        def _unbind_wheel(_):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        return inner

    def _section(self, parent, text):
        lf = ttk.LabelFrame(parent, text=text, padding=6)
        lf.pack(fill="x", pady=4)
        return lf

    def _slider(self, parent, label, var, lo, hi, resolution):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=14).pack(side="left")
        val = ttk.Label(row, width=6,
                        text=self._fmt(var.get(), resolution))
        val.pack(side="right")
        s = tk.Scale(row, from_=lo, to=hi, resolution=resolution,
                     orient="horizontal", variable=var, showvalue=False,
                     command=lambda _v: (val.config(text=self._fmt(var.get(), resolution)),
                                         self._on_change()))
        s.pack(side="left", fill="x", expand=True)
        return s

    @staticmethod
    def _fmt(value, resolution):
        return f"{value:.0f}" if resolution >= 1 else f"{value:.2f}"

    def _build_controls(self, parent):
        # --- image ---
        sec = self._section(parent, "Image")
        ttk.Button(sec, text="Load Image...", command=self.load_image).pack(fill="x")
        ttk.Label(sec, textvariable=self.image_path, wraplength=240,
                  foreground="#555").pack(fill="x", pady=2)

        # --- workspace ranges ---
        sec = self._section(parent, "Workspace ranges (page / robot units)")
        self._range_row(sec, "X range", "x_min", "x_max")
        self._range_row(sec, "Y range", "y_min", "y_max")
        self._range_row(sec, "Z range (dot depth)", "z_min", "z_max")

        # --- mode ---
        sec = self._section(parent, "Mode")
        for text, val in [("Dark (black pen, ink dark areas)", "dark"),
                          ("Light (black pen, ink light areas)", "light"),
                          ("RGB (3 colored-pen passes)", "rgb")]:
            ttk.Radiobutton(sec, text=text, value=val, variable=self.v["mode"],
                            command=self._on_change).pack(anchor="w")
        ttk.Checkbutton(sec, text="Invert tone->ink direction",
                        variable=self.v["invert"], command=self._on_change).pack(anchor="w")

        # --- halftone knobs ---
        sec = self._section(parent, "Halftone knobs")
        self._slider(sec, "Dot pitch", self.v["dot_pitch"], 0.5, 15.0, 0.1)
        self._slider(sec, "Cutoff", self.v["cutoff"], 0.0, 1.0, 0.01)
        self._slider(sec, "Gamma", self.v["gamma"], 0.2, 4.0, 0.05)
        self._slider(sec, "Max dot size", self.v["max_dot"], 0.0, 15.0, 0.1)
        self._slider(sec, "Min dot size", self.v["min_dot"], 0.0, 15.0, 0.1)
        ttk.Label(sec, text="Dot sizes in page units; max 0 = full pitch "
                            "(capped to pitch so dots don't overlap).",
                  wraplength=240, foreground="#555").pack(fill="x", pady=2)
        self._slider(sec, "Min coverage", self.v["min_coverage"], 0.0, 0.5, 0.01)

        # --- adaptive density + budget ---
        sec = self._section(parent, "Dot reduction")
        ttk.Checkbutton(sec, text="Adaptive density (detail-aware)",
                        variable=self.v["adaptive"], command=self._on_change).pack(anchor="w")
        self._slider(sec, "  fine pitch", self.v["fine_pitch"], 0.3, 8.0, 0.1)
        self._slider(sec, "  detail thresh", self.v["detail_threshold"], 0.02, 0.5, 0.01)
        ttk.Checkbutton(sec, text="Limit total dots (budget)",
                        variable=self.v["budget_on"], command=self._on_change).pack(anchor="w")
        self._slider(sec, "  max dots", self.v["max_dots"], 200, 20000, 100)

        # --- Z discretization ---
        sec = self._section(parent, "Z discretization")
        ttk.Checkbutton(sec, text="Discretize Z into integer levels",
                        variable=self.v["z_levels_on"], command=self._on_change).pack(anchor="w")
        self._slider(sec, "  min level", self.v["z_level_min"], 0, 20, 1)
        self._slider(sec, "  N (top level)", self.v["z_levels"], 1, 20, 1)
        ttk.Label(sec, text="Z = integer level; coverage split into equal bands "
                            "from min level up to N.",
                  wraplength=240, foreground="#555").pack(fill="x", pady=2)

        # --- actions ---
        sec = self._section(parent, "Output")
        row = ttk.Frame(sec); row.pack(fill="x", pady=1)
        ttk.Label(row, text="Prefix", width=8).pack(side="left")
        ttk.Entry(row, textvariable=self.v["out_prefix"]).pack(side="left", fill="x", expand=True)
        ttk.Checkbutton(sec, text="Auto-update preview",
                        variable=self.v["auto_update"]).pack(anchor="w", pady=2)
        ttk.Button(sec, text="Update preview", command=self.update_preview).pack(fill="x", pady=1)
        ttk.Button(sec, text="Export CSV(s)", command=self.export).pack(fill="x", pady=1)
        ttk.Button(sec, text="Save preview PNG...", command=self.save_png).pack(fill="x", pady=1)

    def _range_row(self, parent, label, lo_key, hi_key):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=16).pack(side="left")
        e1 = ttk.Entry(row, textvariable=self.v[lo_key], width=7)
        e1.pack(side="left", padx=2)
        ttk.Label(row, text="to").pack(side="left")
        e2 = ttk.Entry(row, textvariable=self.v[hi_key], width=7)
        e2.pack(side="left", padx=2)
        for e in (e1, e2):
            e.bind("<Return>", lambda _e: self._on_change())
            e.bind("<FocusOut>", lambda _e: self._on_change())

    # ------------------------------------------------------------------ #
    # Config assembly
    # ------------------------------------------------------------------ #
    def _build_config(self) -> hr.Config:
        v = self.v
        return hr.Config(
            image_path=self.image_path.get(),
            x_range=(v["x_min"].get(), v["x_max"].get()),
            y_range=(v["y_min"].get(), v["y_max"].get()),
            z_range=(v["z_min"].get(), v["z_max"].get()),
            mode=v["mode"].get(),
            dot_pitch=v["dot_pitch"].get(),
            cutoff=v["cutoff"].get(),
            gamma=v["gamma"].get(),
            max_dot=v["max_dot"].get(),
            min_dot=v["min_dot"].get(),
            min_coverage=v["min_coverage"].get(),
            invert=v["invert"].get(),
            adaptive=v["adaptive"].get(),
            fine_pitch=v["fine_pitch"].get(),
            detail_threshold=v["detail_threshold"].get(),
            max_dots=v["max_dots"].get() if v["budget_on"].get() else 0,
            z_levels=v["z_levels"].get() if v["z_levels_on"].get() else 0,
            z_level_min=v["z_level_min"].get(),
            out_prefix=v["out_prefix"].get(),
            show_plot=False,
        )

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                       ("All files", "*.*")])
        if path:
            self.image_path.set(path)
            self.update_preview()

    def _on_change(self, *_):
        if self.v["auto_update"].get() and self.image_path.get():
            self.update_preview()

    def update_preview(self):
        if not self.image_path.get():
            self.status.config(text="Load an image first.")
            return
        try:
            cfg = self._build_config()
            dots, layout, max_dot = hr.compute_dots(cfg)
            self._last_result = (cfg, dots, layout, max_dot)
            hr.render_dots(self.ax, cfg, dots, layout, max_dot)
            self.canvas.draw_idle()
            ds = [d.diameter for d in dots]
            rng = f"{min(ds):.2f}-{max(ds):.2f}" if ds else "-"
            self.status.config(
                text=(f"{layout['ncols']}x{layout['nrows']} grid | {len(dots)} dots | "
                      f"diam {rng} | max dot {max_dot:.2f}"))
        except Exception as exc:
            traceback.print_exc()
            self.status.config(text=f"Error: {exc}")

    def export(self):
        if not self._last_result:
            messagebox.showinfo("Export", "Generate a preview first.")
            return
        cfg, dots, _, _ = self._last_result
        out_dir = filedialog.askdirectory(title="Choose output folder")
        if not out_dir:
            return
        cfg.out_prefix = os.path.join(out_dir, self.v["out_prefix"].get())
        written = hr.write_outputs(cfg, dots)
        msg = "\n".join(f"{name}: {n} dots -> {os.path.basename(path)}"
                        for name, path, n in written)
        messagebox.showinfo("Export complete", msg)
        self.status.config(text=f"Exported {len(written)} file(s) to {out_dir}")

    def save_png(self):
        if not self._last_result:
            messagebox.showinfo("Save PNG", "Generate a preview first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save preview PNG", defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            self.status.config(text=f"Saved preview -> {path}")


def main():
    root = tk.Tk()
    HalftoneGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
