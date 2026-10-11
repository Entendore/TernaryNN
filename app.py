#!/usr/bin/env python3
"""
Ternary Neural Network (TNN) Playground 4.0 — PySide6 + numpy.

Core  : forward pass on ternary weights Ŵ ∈ {−1, 0, +1} (Δ = factor·mean|W|),
        straight-through-estimator training of latent weights (SGD+momentum/Adam).
Practice: threshold tuning + ROC/AUC, confusion/precision/recall/F1, early
        stopping w/ best-epoch rollback, LR range test, K-fold cross-validation,
        PTQ-vs-QAT comparison, experiment log (CSV), jitter augmentation,
        layer freezing (greedy layer-wise), class imbalance.
Hands-on: click neurons to inspect, 🎯-pick samples to trace, click weight
        matrix cells to hand-edit ternary weights live, challenge badges.
Deploy: memory/ops cost panel + MCU-ready C export.

Run:  python tnn_playground.py
"""

import json
import math
import random
import sys
from datetime import datetime

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                           QPolygonF, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPushButton,
                               QScrollArea, QSlider, QSpinBox, QSplitter,
                               QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

# --------------------------------------------------------------- palette ---
C_BLUE, C_ORANGE = QColor("#4285f4"), QColor("#ff8c00")
C_RED, C_GREEN = QColor("#d93025"), QColor("#188038")


class TH:
    dark = False

    @classmethod
    def refresh(cls):
        if cls.dark:
            cls.bg = QColor("#17181b"); cls.text = QColor("#e8eaed")
            cls.muted = QColor("#9aa0a6"); cls.grid = QColor(255, 255, 255, 26)
            cls.frame = QColor(255, 255, 255, 55); cls.zero_cell = QColor(255, 255, 255, 38)
            cls.hover_box = QColor(45, 47, 51, 240)
            cls.heat_bg = (23, 24, 27)
            cls.heat_blue = (138, 180, 248); cls.heat_orange = (255, 167, 38)
        else:
            cls.bg = QColor("#ffffff"); cls.text = QColor("#202124")
            cls.muted = QColor("#80868b"); cls.grid = QColor(0, 0, 0, 26)
            cls.frame = QColor(0, 0, 0, 60); cls.zero_cell = QColor(0, 0, 0, 36)
            cls.hover_box = QColor(255, 255, 255, 240)
            cls.heat_bg = (250, 250, 251)
            cls.heat_blue = (66, 133, 244); cls.heat_orange = (255, 140, 0)


TH.refresh()


def small_font(w, bold=False):
    f = QFont(w.font())
    ps = w.font().pointSize()
    f.setPointSize(max(7, int(ps * 0.85)) if ps > 0 else 8)
    f.setBold(bold)
    return f


# --------------------------------------------------------------- datasets ---
def dataset_xor(n, rng):
    X = rng.uniform(-1.0, 1.0, (n, 2))
    return X, (X[:, 0] * X[:, 1] > 0).astype(float).reshape(-1, 1)


def dataset_circle(n, rng):
    n1 = n // 2
    a = rng.uniform(0, 2 * np.pi, n1)
    r = np.sqrt(rng.uniform(0.0, 0.45 ** 2, n1))
    inner = np.column_stack([r * np.cos(a), r * np.sin(a)])
    a = rng.uniform(0, 2 * np.pi, n - n1)
    r = rng.uniform(0.62, 1.0, n - n1)
    outer = np.column_stack([r * np.cos(a), r * np.sin(a)])
    return (np.vstack([inner, outer]),
            np.vstack([np.ones((n1, 1)), np.zeros((n - n1, 1))]))


def dataset_spiral(n, rng):
    m = n // 2
    X, y = [], []
    for cls in (0, 1):
        j = np.arange(m)
        r = (j + 1) / m * 0.92
        t = 1.75 * 2.0 * np.pi * j / m + cls * np.pi
        X.append(np.column_stack([r * np.sin(t), r * np.cos(t)]))
        y.append(np.full((m, 1), float(cls)))
    return np.vstack(X), np.vstack(y)


def dataset_moons(n, rng):
    m = n // 2
    t = rng.uniform(0, np.pi, m)
    A = np.column_stack([np.cos(t), np.sin(t)])
    t = rng.uniform(0, np.pi, n - m)
    B = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X = (np.vstack([A, B]) - np.array([0.5, 0.25])) * np.array([0.62, 0.85])
    return X, np.vstack([np.zeros((m, 1)), np.ones((n - m, 1))])


def dataset_gauss(n, rng):
    m = n // 2
    A = rng.normal([0.45, 0.45], 0.20, (m, 2))
    B = rng.normal([-0.45, -0.45], 0.20, (n - m, 2))
    return (np.vstack([A, B]),
            np.vstack([np.ones((m, 1)), np.zeros((n - m, 1))]))


def dataset_stripes(n, rng):
    X = rng.uniform(-1.0, 1.0, (n, 2))
    y = (np.floor((X[:, 0] + 1.0) / 0.5) % 2 == 0).astype(float).reshape(-1, 1)
    return X, y


def dataset_checker(n, rng):
    X = rng.uniform(-1.0, 1.0, (n, 2))
    y = ((np.floor((X[:, 0] + 1.0) / 0.5) + np.floor((X[:, 1] + 1.0) / 0.5)) % 2 == 0)
    return X, y.astype(float).reshape(-1, 1)


def dataset_sine(n, rng):
    X = rng.uniform(-1.0, 1.0, (n, 2))
    y = (np.sin(2.5 * X[:, 0]) + 0.4 * X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


DATASETS = {"XOR": dataset_xor, "Circle": dataset_circle, "Spiral": dataset_spiral,
            "Moons": dataset_moons, "Gaussians": dataset_gauss,
            "Stripes": dataset_stripes, "Checker": dataset_checker, "Sine": dataset_sine}


def generate_data(kind, n, sigma, rng):
    X, y = DATASETS[kind](n, rng)
    if sigma > 0:
        X = X + rng.normal(0.0, sigma, X.shape)
    X = np.clip(X, -1.05, 1.05)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def subsample_imbalance(X, y, n, f1, rng):
    i1 = np.where(y[:, 0] > 0.5)[0]
    i0 = np.where(y[:, 0] <= 0.5)[0]
    n1 = min(int(round(n * f1)), len(i1))
    n0 = min(n - n1, len(i0))
    pick = rng.permutation(np.concatenate([
        rng.choice(i1, n1, replace=False), rng.choice(i0, n0, replace=False)]))
    return X[pick], y[pick]


# --------------------------------------------------------------- features ---
FEATURES = ["x₁", "x₂", "x₁²", "x₂²", "x₁x₂", "sin x₁", "sin x₂"]


def featurize(X, active):
    cols = []
    for i in active:
        if i == 0:   cols.append(X[:, 0:1])
        elif i == 1: cols.append(X[:, 1:2])
        elif i == 2: cols.append(X[:, 0:1] ** 2)
        elif i == 3: cols.append(X[:, 1:2] ** 2)
        elif i == 4: cols.append((X[:, 0] * X[:, 1])[:, None])
        elif i == 5: cols.append(np.sin(X[:, 0:1]))
        else:        cols.append(np.sin(X[:, 1:2]))
    return np.hstack(cols)


# ------------------------------------------------------------- TNN model ---
class TernaryNet:
    def __init__(self, sizes, activation="tanh", ternary=True, optimizer="sgd",
                 delta_factor=0.7, quantize_bias=False):
        rng = np.random.default_rng()
        self.sizes = list(sizes)
        self.activation = activation
        self.ternary = bool(ternary)
        self.optimizer = optimizer
        self.delta_factor = float(delta_factor)
        self.quantize_bias = bool(quantize_bias)
        self.L = len(sizes) - 1
        self.t = 0
        self.last_gnorms = []
        self.frozen = [False] * self.L
        self.W, self.b = [], []
        self.vW, self.vb, self.mW, self.vV, self.mb, self.vB = [], [], [], [], [], []
        for fi, fo in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0.0, math.sqrt(1.0 / fi), (fi, fo)))
            self.b.append(np.zeros(fo))
            self.vW.append(np.zeros((fi, fo))); self.vb.append(np.zeros(fo))
            self.mW.append(np.zeros((fi, fo))); self.vV.append(np.zeros((fi, fo)))
            self.mb.append(np.zeros(fo));       self.vB.append(np.zeros(fo))

    def snapshot(self):
        return ([w.copy() for w in self.W], [b.copy() for b in self.b], int(self.t))

    def restore(self, s):
        for l in range(self.L):
            self.W[l][:] = s[0][l]
            self.b[l][:] = s[1][l]
        self.t = int(s[2])

    def clone(self):
        n = TernaryNet(self.sizes, self.activation, ternary=self.ternary,
                       optimizer=self.optimizer, delta_factor=self.delta_factor,
                       quantize_bias=self.quantize_bias)
        n.restore(self.snapshot())
        return n

    # ---- quantisation ------------------------------------------------------
    def _q(self, arr):
        d = self.delta_factor * float(np.mean(np.abs(arr)))
        return np.where(np.abs(arr) <= d, 0.0, np.sign(arr)), d

    def quantized(self):
        return [self._q(w)[0] for w in self.W] if self.ternary else self.W

    def biases_q(self):
        if not self.ternary or not self.quantize_bias:
            return self.b
        return [self._q(b)[0] for b in self.b]

    def deltas(self):
        return [self._q(w)[1] for w in self.W] if self.ternary else [None] * self.L

    def zero_fraction(self):
        q = self.quantized()
        tot = sum(w.size for w in q)
        z = sum(int(np.count_nonzero(w == 0.0)) for w in q)
        return z, tot

    def ops(self):
        Wq, Bq = self.quantized(), self.biases_q()
        adds = macs = 0
        for W, b in zip(Wq, Bq):
            adds += int(np.count_nonzero(W)) + W.shape[1]
            macs += W.size + W.shape[1]
        return adds, macs, (100.0 * adds / macs if macs else 0.0)

    # ---- activations ---------------------------------------------------------
    def _act(self, Z):
        return np.maximum(Z, 0.0) if self.activation == "ReLU" else np.tanh(Z)

    def _agrad(self, A, Z):
        return (Z > 0.0).astype(float) if self.activation == "ReLU" else 1.0 - A * A

    # ---- forward / backward ----------------------------------------------------
    def forward(self, X):
        Wq, Bq = self.quantized(), self.biases_q()
        acts, Zs, A = [X], [], X
        for l, W in enumerate(Wq):
            Z = A @ W + Bq[l]
            Zs.append(Z)
            A = (1.0 / (1.0 + np.exp(-np.clip(Z, -30, 30))) if l == self.L - 1
                 else self._act(Z))
            acts.append(A)
        return A, acts, Zs, Wq, Bq

    def predict(self, X):
        return self.forward(X)[0]

    def trace(self, X):
        return self.forward(X)[1]

    @staticmethod
    def loss(P, Y):
        P = np.clip(P, 1e-7, 1 - 1e-7)
        return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))

    def train_batch(self, Xb, Yb, lr, l2=0.0):
        P, acts, Zs, Wq, _ = self.forward(Xb)
        delta = (P - Yb) / len(Xb)
        self.t += 1
        gnorms = []
        for l in reversed(range(self.L)):
            dW = acts[l].T @ delta
            db = delta.sum(axis=0)
            gnorms.append(float(np.linalg.norm(dW)))
            if l > 0:                       # grads must flow through frozen layers
                delta = (delta @ Wq[l].T) * self._agrad(acts[l], Zs[l - 1])
            if self.frozen[l]:
                continue
            if l2:
                dW = dW + l2 * self.W[l]
            if self.optimizer == "adam":
                b1, b2, eps = 0.9, 0.999, 1e-8
                self.mW[l] = b1 * self.mW[l] + (1 - b1) * dW
                self.vV[l] = b2 * self.vV[l] + (1 - b2) * dW * dW
                self.mb[l] = b1 * self.mb[l] + (1 - b1) * db
                self.vB[l] = b2 * self.vB[l] + (1 - b2) * db * db
                self.W[l] -= lr * (self.mW[l] / (1 - b1 ** self.t)) / \
                    (np.sqrt(self.vV[l] / (1 - b2 ** self.t)) + eps)
                self.b[l] -= lr * (self.mb[l] / (1 - b1 ** self.t)) / \
                    (np.sqrt(self.vB[l] / (1 - b2 ** self.t)) + eps)
            else:
                self.vW[l] = 0.9 * self.vW[l] - lr * dW
                self.vb[l] = 0.9 * self.vb[l] - lr * db
                self.W[l] += self.vW[l]
                self.b[l] += self.vb[l]
            np.clip(self.W[l], -1.0, 1.0, out=self.W[l])
        self.last_gnorms = gnorms[::-1]
        return self.loss(P, Yb)


# -------------------------------------------------------- layer editor -----
class LayerEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = [4, 4]
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(6)
        self.rebuild()

    def rebuild(self):
        while self.lay.count():
            it = self.lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for i, n in enumerate(self.layers):
            chip = QFrame()
            chip.setObjectName("chip")
            v = QVBoxLayout(chip)
            v.setContentsMargins(7, 4, 7, 4)
            v.setSpacing(3)
            t = QLabel(f"h{i + 1}\n{n}u")
            t.setAlignment(Qt.AlignCenter)
            h = QHBoxLayout(); h.setSpacing(2)
            bm = QPushButton("−"); bp = QPushButton("+"); bx = QPushButton("×")
            for b in (bm, bp, bx):
                b.setObjectName("mini"); b.setFixedWidth(22); b.setFocusPolicy(Qt.NoFocus)
                h.addWidget(b)
            bm.clicked.connect(lambda _, k=i: self._adj(k, -1))
            bp.clicked.connect(lambda _, k=i: self._adj(k, +1))
            bx.clicked.connect(lambda _, k=i: self._del(k))
            v.addWidget(t); v.addLayout(h)
            self.lay.addWidget(chip)
        add = QPushButton("+ layer")
        add.setObjectName("mini"); add.setFocusPolicy(Qt.NoFocus)
        add.setEnabled(len(self.layers) < 4)
        add.clicked.connect(self._add)
        self.lay.addWidget(add)
        self.lay.addStretch(1)

    def _adj(self, i, d):
        self.layers[i] = int(np.clip(self.layers[i] + d, 1, 8))
        self.rebuild(); self.changed.emit()

    def _del(self, i):
        if len(self.layers) <= 1:
            return
        del self.layers[i]
        self.rebuild(); self.changed.emit()

    def _add(self):
        if len(self.layers) < 4:
            self.layers.append(4)
            self.rebuild(); self.changed.emit()

    def set_layers(self, ls):
        ls = [int(x) for x in ls][:4]
        self.layers = ls if ls else [4]
        self.rebuild()


# ------------------------------------------------------------ canvases -----
class BoundaryCanvas(QWidget):
    GRID, RANGE = 140, 1.1
    samplePicked = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        self.net = None
        self.Xtr = self.Ytr = self.Xte = self.Yte = None
        self._probe = self._grid = None
        self._hover = self._hover_p = None
        self.show_test = True
        self.discretize = False
        self.pick_mode = False
        self.sel_pt = None
        self.cx = self.cy = 0.0
        self.half = self.RANGE
        self._pan = False
        self._last = None

    def set_state(self, net, Xtr, Ytr, Xte, Yte, probe, grid):
        self.net, self.Xtr, self.Ytr, self.Xte, self.Yte = net, Xtr, Ytr, Xte, Yte
        self._probe, self._grid = probe, grid
        self.update()

    def set_flags(self, show_test, discretize):
        self.show_test, self.discretize = show_test, discretize
        self.update()

    def set_pick_mode(self, on):
        self.pick_mode = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def set_selected(self, pt):
        self.sel_pt = pt
        self.update()

    def reset_view(self):
        self.cx = self.cy = 0.0
        self.half = self.RANGE
        self.update()

    def _plot_rect(self):
        m, lg = 12.0, 26.0
        aw, ah = self.width() - 2 * m, self.height() - m - lg - m
        side = max(80.0, min(aw, ah))
        return QRectF(m + (aw - side) / 2, m + (ah - side) / 2, side, side)

    def _hy(self, r):
        return self.half * r.height() / r.width()

    def _map(self, r, x, y):
        return QPointF(r.center().x() + (x - self.cx) / (2 * self.half) * r.width(),
                       r.center().y() - (y - self.cy) / (2 * self._hy(r)) * r.height())

    def _unmap(self, r, px, py):
        return (self.cx + (px - r.center().x()) / r.width() * 2 * self.half,
                self.cy - (py - r.center().y()) / r.height() * 2 * self._hy(r))

    def wheelEvent(self, e):
        r = self._plot_rect()
        if self.net is None or not r.contains(e.position()):
            return
        dx, dy = self._unmap(r, e.position().x(), e.position().y())
        f = 1.25 if e.angleDelta().y() > 0 else 1 / 1.25
        newh = float(np.clip(self.half / f, 0.08, 2.6))
        s = newh / self.half
        self.cx = float(np.clip(dx - (dx - self.cx) * s, -2.5, 2.5))
        self.cy = float(np.clip(dy - (dy - self.cy) * s, -2.5, 2.5))
        self.half = newh
        self.update()

    def mousePressEvent(self, e):
        r = self._plot_rect()
        if e.button() != Qt.LeftButton or self.net is None or not r.contains(e.position()):
            return
        if self.pick_mode:
            best, bd = None, 1e9
            for test, X, Y in ((False, self.Xtr, self.Ytr), (True, self.Xte, self.Yte)):
                if X is None:
                    continue
                for i, ((x, y), lab) in enumerate(zip(X, Y)):
                    d = (self._map(r, float(x), float(y)) - e.position()).manhattanLength()
                    if d < bd:
                        bd, best = d, dict(x=float(x), y=float(y), label=float(lab),
                                           test=test, idx=i)
            self.samplePicked.emit(best if (best and bd < 40) else None)
            return
        self._pan = True
        self._last = e.position()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, e):
        self._pan = False
        self.setCursor(Qt.CrossCursor if self.pick_mode else Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, e):
        self.reset_view()

    def mouseMoveEvent(self, e):
        r = self._plot_rect()
        if self._pan:
            dx = (e.position().x() - self._last.x()) / r.width() * 2 * self.half
            dy = (e.position().y() - self._last.y()) / r.height() * 2 * self._hy(r)
            self.cx = float(np.clip(self.cx - dx, -2.5, 2.5))
            self.cy = float(np.clip(self.cy + dy, -2.5, 2.5))
            self._last = e.position()
            self.update()
            return
        if self.net is not None and r.contains(e.position()):
            x, y = self._unmap(r, e.position().x(), e.position().y())
            self._hover = (x, y)
            self._hover_p = self._probe(np.array([[x, y]])) if self._probe else None
            self.update()

    def leaveEvent(self, e):
        if self._hover is not None:
            self._hover = self._hover_p = None
            self.update()

    def _heat_image(self, r):
        G = self.GRID
        hx, hy = self.half, self._hy(r)
        xs = np.linspace(self.cx - hx, self.cx + hx, G)
        ys = np.linspace(self.cy + hy, self.cy - hy, G)
        XX, YY = np.meshgrid(xs, ys)
        P = self._grid(np.column_stack([XX.ravel(), YY.ravel()])).reshape(G, G)
        blue, orange = np.array(TH.heat_blue, float), np.array(TH.heat_orange, float)
        base = np.array(TH.heat_bg, float)
        t = np.clip(np.abs(P - 0.5) * 2, 0, 1)[..., None]
        if self.discretize:
            t = np.ones_like(t)
        fg = np.where((P > 0.5)[..., None], orange, blue)
        col = np.clip(base * (1 - t) + fg * t, 0, 255).astype(np.uint8)
        buf = np.ascontiguousarray(col)
        return QImage(buf.data, G, G, 3 * G, QImage.Format_RGB888).copy()

    @staticmethod
    def _nice_step(hx):
        for s in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
            if hx / s <= 6:
                return s
        return 2.0

    def _points(self, p, r, X, Y, test):
        for (x, y), lab in zip(X, Y):
            c = C_ORANGE if lab > 0.5 else C_BLUE
            pos = self._map(r, float(x), float(y))
            p.setBrush(c)
            p.setPen(QPen(c.darker(125), 1.2))
            if test:
                p.drawRect(QRectF(pos.x() - 3.4, pos.y() - 3.4, 6.8, 6.8))
            else:
                p.drawEllipse(pos, 4.2, 4.2)

    def _legend(self, p, r):
        y = r.bottom() + 17
        p.setFont(small_font(self))
        fm = p.fontMetrics()
        x = r.left()
        for kind, col, label in (("dot", C_ORANGE, "class 1"), ("dot", C_BLUE, "class 0"),
                                 ("ring", QColor("#5f6368"), "train"),
                                 ("sq", QColor("#5f6368"), "test")):
            p.setBrush(TH.bg if kind == "ring" else col)
            p.setPen(QPen(col, 1.2))
            if kind == "sq":
                p.drawRect(QRectF(x - 3.4, y - 3.4, 6.8, 6.8))
            else:
                p.drawEllipse(QPointF(x, y), 4.2, 4.2)
            p.setPen(TH.muted)
            p.drawText(QPointF(x + 8, y + 4), label)
            x += 8 + fm.horizontalAdvance(label) + 18

    def _hover_box(self, p, r):
        if self._hover is None:
            return
        pt = self._map(r, *self._hover)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(TH.text, 1.3))
        p.drawEllipse(pt, 7, 7)
        if self._hover_p is None:
            return
        txt = (f"class {int(self._hover_p > 0.5)}  (p = {self._hover_p:.2f})"
               if self.discretize else f"p(class 1) = {self._hover_p:.2f}")
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(txt) + 12, fm.height() + 6
        bx = min(pt.x() + 12, r.right() - w)
        by = max(pt.y() - h - 8, r.top() + 2)
        p.setBrush(TH.hover_box)
        p.setPen(QPen(TH.frame, 1))
        p.drawRoundedRect(QRectF(bx, by, w, h), 4, 4)
        p.setPen(TH.text)
        p.drawText(QPointF(bx + 6, by + h - 7), txt)

    def _scale_chip(self, p, r):
        txt = f"±{self.half:.2f} · scroll=zoom · drag=pan · dbl-click=reset"
        p.setFont(small_font(self))
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(txt) + 10, fm.height() + 4
        p.setBrush(TH.hover_box)
        p.setPen(QPen(TH.frame, 1))
        p.drawRoundedRect(QRectF(r.left() + 6, r.top() + 6, w, h), 4, 4)
        p.setPen(TH.text)
        p.drawText(QPointF(r.left() + 11, r.top() + 6 + h - 6), txt)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), TH.bg)
        r = self._plot_rect()
        if self.net is None or self.Xtr is None:
            p.setPen(TH.muted)
            p.drawText(self.rect(), Qt.AlignCenter, "no data")
            return
        p.drawImage(r, self._heat_image(r))
        step = self._nice_step(self.half)
        hy = self._hy(r)
        p.setPen(QPen(TH.grid, 1))
        x = math.ceil((self.cx - self.half) / step) * step
        while x <= self.cx + self.half + 1e-9:
            if abs(x) > 1e-9:
                p.drawLine(self._map(r, x, self.cy - hy), self._map(r, x, self.cy + hy))
            x += step
        y = math.ceil((self.cy - hy) / step) * step
        while y <= self.cy + hy + 1e-9:
            if abs(y) > 1e-9:
                p.drawLine(self._map(r, self.cx - self.half, y),
                           self._map(r, self.cx + self.half, y))
            y += step
        p.setPen(QPen(TH.frame, 1))
        if -hy <= -self.cy <= hy:
            a = self._map(r, self.cx, 0.0)
            p.drawLine(QPointF(r.left(), a.y()), QPointF(r.right(), a.y()))
        if -self.half <= -self.cx <= self.half:
            a = self._map(r, 0.0, self.cy)
            p.drawLine(QPointF(a.x(), r.top()), QPointF(a.x(), r.bottom()))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(TH.frame, 1))
        p.drawRect(r)
        self._points(p, r, self.Xtr, self.Ytr, False)
        if self.show_test:
            self._points(p, r, self.Xte, self.Yte, True)
        if self.sel_pt is not None:
            pt = self._map(r, *self.sel_pt)
            p.setPen(QPen(C_RED, 1.8))
            p.drawEllipse(pt, 7, 7)
            p.drawLine(QPointF(pt.x() - 11, pt.y()), QPointF(pt.x() + 11, pt.y()))
            p.drawLine(QPointF(pt.x(), pt.y() - 11), QPointF(pt.x(), pt.y() + 11))
        self._legend(p, r)
        self._scale_chip(p, r)
        self._hover_box(p, r)


class NetworkCanvas(QWidget):
    neuronClicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(230)
        self.net, self.in_labels = None, []
        self.sel = None
        self.trace = None

    def set_state(self, net, in_labels):
        self.net, self.in_labels = net, in_labels
        if net is not None and self.sel is not None:
            l, j = self.sel
            if l >= len(net.sizes) or j >= net.sizes[l]:
                self.sel = None
        self.update()

    def set_sel(self, sel):
        self.sel = sel
        self.update()

    def set_trace(self, trace):
        self.trace = trace
        self.update()

    def _layout(self):
        W, H = float(self.width()), float(self.height())
        sizes, L, rad = self.net.sizes, len(self.net.sizes), 11.0
        xs = np.linspace(86.0, W - 66.0, L)
        top, bot = 34.0, H - 44.0
        pos = []
        for l, n in enumerate(sizes):
            gap = min(56.0, (bot - top) / max(n, 1)) if n > 1 else 0.0
            y0 = (top + bot) / 2 - gap * (n - 1) / 2
            pos.append([QPointF(xs[l], y0 + k * gap) for k in range(n)])
        return pos, xs, rad

    def mousePressEvent(self, e):
        if self.net is None:
            return
        pos, _, _ = self._layout()
        best, bd = None, 1e9
        for l, col in enumerate(pos):
            for j, pt in enumerate(col):
                d = (pt - e.position()).manhattanLength()
                if d < bd:
                    bd, best = d, (l, j)
        if best and bd < 26:
            self.neuronClicked.emit(*best)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        if self.net is None:
            return
        pos, xs, rad = self._layout()
        W, H = float(self.width()), float(self.height())
        sizes, L = self.net.sizes, len(self.net.sizes)

        ternary = self.net.ternary
        Wq = self.net.quantized()
        for l in range(L - 1):
            for i in range(sizes[l]):
                for j in range(sizes[l + 1]):
                    w = Wq[l][i, j]
                    if ternary:
                        pen = (QPen(C_BLUE, 2.8) if w > 0 else
                               QPen(C_ORANGE, 2.8) if w < 0 else
                               QPen(QColor(255, 255, 255, 40) if TH.dark
                                    else QColor(0, 0, 0, 34), 1.0))
                    else:
                        a = min(1.0, abs(w))
                        if a < 0.02:
                            pen = QPen(TH.zero_cell, 1.0)
                        else:
                            col = QColor(C_BLUE if w > 0 else C_ORANGE)
                            col.setAlpha(int(45 + 200 * a))
                            pen = QPen(col, 0.8 + 2.4 * a)
                    p.setPen(pen)
                    p.drawLine(pos[l][i].x() + rad, pos[l][i].y(),
                               pos[l + 1][j].x() - rad, pos[l + 1][j].y())

        for l in range(L):
            for pt in pos[l]:
                p.setBrush(TH.bg)
                p.setPen(QPen(QColor("#9aa0a6"), 1.4))
                p.drawEllipse(pt, rad, rad)

        if self.trace is not None and len(self.trace) == L:
            for l in range(L):
                for k in range(sizes[l]):
                    v = float(self.trace[l][0, k])
                    if l == L - 1:
                        t, pos_c = abs(v - 0.5) * 2, v > 0.5
                    else:
                        t, pos_c = min(abs(v), 1.0), v > 0
                    if t < 0.04:
                        continue
                    col = QColor(C_BLUE if pos_c else C_ORANGE)
                    col.setAlpha(int(235 * t))
                    p.setBrush(col)
                    p.setPen(Qt.NoPen)
                    p.drawEllipse(pos[l][k], rad - 3.5, rad - 3.5)

        Bq = self.net.biases_q()
        for l in range(L):
            for k, pt in enumerate(pos[l]):
                b = float(Bq[l][k])
                if abs(b) > 1e-6:
                    col = QColor(C_BLUE) if b > 0 else QColor(C_ORANGE)
                    col.setAlpha(int(255 * min(1.0, abs(b) / 1.5)))
                    p.setBrush(col)
                    p.setPen(Qt.NoPen)
                    p.drawEllipse(pt, 5.0, 5.0)

        if self.sel is not None:
            l, j = self.sel
            if l < L and j < sizes[l]:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(C_RED, 2.0))
                p.drawEllipse(pos[l][j], rad + 3.0, rad + 3.0)

        p.setFont(small_font(self))
        fm = p.fontMetrics()
        p.setPen(TH.text)
        for k, pt in enumerate(pos[0]):
            name = self.in_labels[k] if k < len(self.in_labels) else f"x{k}"
            p.drawText(QPointF(8, pt.y() + fm.ascent() / 2 - 1), name)
        p.drawText(pos[-1][0] + QPointF(rad + 8, 4), "ŷ")

        p.setPen(TH.muted)
        act = self.net.activation
        for l in range(L):
            txt = "input" if l == 0 else ("out · σ" if l == L - 1 else f"h{l} · {act}")
            p.drawText(QPointF(xs[l] - fm.horizontalAdvance(txt) / 2, H - 14), txt)

        x, y = 10.0, 16.0
        items = (("+1", C_BLUE, 3.0), ("−1", C_ORANGE, 3.0),
                 ("0", QColor(255, 255, 255, 90) if TH.dark else QColor(0, 0, 0, 70), 1.2))
        if not ternary:
            items = (("|w|>0", C_BLUE, 2.0), ("|w|≈0", QColor(0, 0, 0, 70), 1.2))
        for label, col, wd in items:
            p.setPen(QPen(col, wd))
            p.drawLine(QPointF(x, y), QPointF(x + 16, y))
            p.setPen(TH.text)
            p.drawText(QPointF(x + 20, y + 4), label)
            x += 20 + fm.horizontalAdvance(label) + 12
        if ternary:
            z, tot = self.net.zero_fraction()
            txt = f"{z}/{tot} weights are 0 ({z / tot * 100:.0f}%)" if tot else ""
        else:
            txt = "full-precision reference"
        p.setPen(TH.muted)
        p.drawText(QPointF(W - fm.horizontalAdvance(txt) - 10, y + 4), txt)


class LossCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.tl, self.vl = [], []
        self.ftl = self.fvl = None
        self.mode = "Loss"
        self.best_ep = None
        self.c1, self.c2 = QColor("#1a73e8"), QColor("#e8710a")

    def set_series(self, tl, vl, ftl, fvl, mode, best_ep=None):
        self.tl, self.vl, self.ftl, self.fvl, self.mode = tl, vl, ftl, fvl, mode
        self.best_ep = best_ep
        self.update()

    def _fmt(self, v):
        return f"{v * 100:.0f}%" if self.mode == "Accuracy" else f"{v:.3f}"

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        if len(self.tl) < 2:
            p.setPen(TH.muted)
            p.drawText(self.rect(), Qt.AlignCenter, "press ▶ Train")
            return
        area = QRectF(46, 12, self.width() - 56, self.height() - 40)
        vals = list(self.tl) + list(self.vl)
        if self.ftl:
            vals += list(self.ftl) + list(self.fvl)
        if self.mode == "Accuracy":
            lo, hi = 0.0, 1.0
        else:
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                hi = lo + 1e-9
            pad = (hi - lo) * 0.12
            lo, hi = lo - pad, hi + pad
        n = len(self.tl)
        ym = lambda v: area.bottom() - (v - lo) / (hi - lo) * area.height()
        p.setFont(small_font(self))
        for k in range(4):
            v = lo + (hi - lo) * k / 3
            p.setPen(QPen(TH.grid, 1))
            p.drawLine(QPointF(area.left(), ym(v)), QPointF(area.right(), ym(v)))
            p.setPen(TH.muted)
            p.drawText(QPointF(6, ym(v) + 4), self._fmt(v))
        xm = lambda i: area.left() + area.width() * i / (n - 1)

        if self.best_ep and 1 <= self.best_ep <= n:
            bx = xm(self.best_ep - 1)
            p.setPen(QPen(C_GREEN, 1.4, Qt.DashLine))
            p.drawLine(QPointF(bx, area.top()), QPointF(bx, area.bottom()))
            p.setPen(C_GREEN)
            p.drawText(QPointF(bx + 3, area.top() + 9), "best")

        series = [(self.tl, self.c1, False), (self.vl, self.c2, False)]
        if self.ftl:
            series += [(self.ftl, self.c1, True), (self.fvl, self.c2, True)]
        fm = p.fontMetrics()
        for data, col, dashed in series:
            p.setPen(QPen(col, 1.8, Qt.DashLine if dashed else Qt.SolidLine))
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(xm(i), ym(v)) for i, v in enumerate(data)]))
            if data:
                p.setPen(col)
                p.drawText(QPointF(min(xm(n - 1) + 5, area.right() - 4),
                                   ym(data[-1]) - 4), self._fmt(data[-1]))

        lx, ly = area.left() + 8, area.top() + 10
        leg = [("train", self.c1, False), ("test", self.c2, False)]
        if self.ftl:
            leg += [("train FP", self.c1, True), ("test FP", self.c2, True)]
        for label, col, dashed in leg:
            p.setPen(QPen(col, 2, Qt.DashLine if dashed else Qt.SolidLine))
            p.drawLine(QPointF(lx, ly), QPointF(lx + 14, ly))
            p.setPen(TH.muted)
            p.drawText(QPointF(lx + 18, ly + 4), label)
            lx += 18 + fm.horizontalAdvance(label) + 14
        p.drawText(QPointF(area.right() - fm.horizontalAdvance("epoch"),
                           self.height() - 8), "epoch")


class HistCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.latent, self.delta, self.counts = None, None, (0, 0, 0)

    def set_data(self, latent, delta, counts):
        self.latent, self.delta, self.counts = latent, delta, counts
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        p.setFont(small_font(self))
        fm = p.fontMetrics()
        m1, z0, p1 = self.counts
        gray = QColor("#9aa0a6")
        x = 14.0
        for txt, col in ((f"−1 × {m1}", C_ORANGE), (f"0 × {z0}", gray),
                         (f"+1 × {p1}", C_BLUE),
                         (f"Δ = ±{self.delta:.2f}" if self.delta is not None else "", C_RED)):
            if not txt:
                continue
            p.setPen(col)
            p.drawText(QPointF(x, 18), txt)
            x += fm.horizontalAdvance(txt) + 16
        if self.latent is None or len(self.latent) == 0:
            return
        area = QRectF(14, 30, self.width() - 28, self.height() - 44)
        bins = 27
        cnt, _ = np.histogram(self.latent, bins=bins, range=(-1.0, 1.0))
        mx = max(int(cnt.max()), 1)
        bw = area.width() / bins
        base_y = area.bottom()
        p.setPen(QPen(TH.frame, 1))
        p.drawLine(QPointF(area.left(), base_y), QPointF(area.right(), base_y))
        zx = area.left() + area.width() / 2
        p.setPen(QPen(TH.frame, 1, Qt.DashLine))
        p.drawLine(QPointF(zx, area.top()), QPointF(zx, base_y))
        p.setPen(Qt.NoPen)
        for i, c in enumerate(cnt):
            if c <= 0:
                continue
            center = -1.0 + (i + 0.5) * 2.0 / bins
            if self.delta is not None:
                col = gray if abs(center) <= self.delta else \
                    (C_BLUE if center > 0 else C_ORANGE)
                col = QColor(col); col.setAlpha(150)
            else:
                col = QColor("#8a8f98"); col.setAlpha(150)
            p.setBrush(col)
            h = c / mx * (area.height() - 4)
            p.drawRect(QRectF(area.left() + i * bw + 0.5, base_y - h,
                              max(bw - 1.2, 1.0), h))
        if self.delta is not None and 0 < self.delta < 1:
            for sgn in (-1, 1):
                xd = area.left() + (sgn * self.delta + 1) / 2 * area.width()
                p.setPen(QPen(C_RED, 1.5, Qt.DashLine))
                p.drawLine(QPointF(xd, area.top() + 2), QPointF(xd, base_y))


class MatrixCanvas(QWidget):
    cellClicked = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.net, self.layer = None, -1
        self._cells = []

    def set_data(self, net, layer):
        self.net, self.layer = net, layer
        self.update()

    def mousePressEvent(self, e):
        for l, i, j, r in self._cells:
            if r.contains(e.position()):
                self.cellClicked.emit(l, i, j)
                return

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        p.setFont(small_font(self))
        self._cells = []
        if self.net is None:
            return
        mats = self.net.quantized() if self.net.ternary else self.net.W
        lats = self.net.W
        sel = list(range(len(mats))) if self.layer < 0 else [self.layer]
        layers_list = [(mats[l], lats[l], l) for l in sel]
        rows = max(w.shape[0] for w, _, _ in layers_list)
        gap = 18.0
        cols = sum(w.shape[1] for w, _, _ in layers_list) + gap * (len(layers_list) - 1)
        m = 14.0
        area = QRectF(m, 30, self.width() - 2 * m, self.height() - 30 - m - 16)
        cs = float(np.clip(min(area.width() / max(cols, 1), area.height() / max(rows, 1)),
                           3.0, 24.0))
        x0 = area.left() + (area.width() - cols * cs) / 2
        y0 = area.top() + (area.height() - rows * cs) / 2
        fm = p.fontMetrics()
        x = x0
        for w, wl, lidx in layers_list:
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    val, lat = w[i, j], wl[i, j]
                    r = QRectF(x + j * cs, y0 + i * cs, max(cs - 0.6, 1.0),
                               max(cs - 0.6, 1.0))
                    if self.net.ternary:
                        if val > 0:
                            p.fillRect(r, C_BLUE)
                        elif val < 0:
                            p.fillRect(r, C_ORANGE)
                        else:
                            p.fillRect(r, TH.zero_cell)
                        if cs >= 6:
                            self._cells.append((lidx, i, j, r))
                    else:
                        a = min(1.0, abs(lat))
                        if a < 0.02:
                            p.fillRect(r, TH.zero_cell)
                        else:
                            col = QColor(C_BLUE if lat > 0 else C_ORANGE)
                            col.setAlpha(int(35 + 210 * a))
                            p.fillRect(r, col)
            if cs >= 9:
                p.setPen(TH.muted)
                p.drawText(QPointF(x, 20), f"L{lidx + 1}")
            x += w.shape[1] * cs + gap
        p.setPen(TH.muted)
        hint = ("rows = inputs · cols = units · click a cell to cycle +1 / 0 / −1"
                if self.net.ternary else "α = |latent w| · editing available in ternary view")
        p.drawText(QPointF(14, self.height() - 6), hint)


class NeuronCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.net, self.sel = None, None
        self.in_labels, self.active = [], []
        self.epoch = 0
        self._cache_key, self._cache_img = None, None

    def set_data(self, net, sel, in_labels, active, epoch):
        self.net, self.sel = net, sel
        self.in_labels, self.active, self.epoch = in_labels, active, epoch
        if net is not None and sel is not None:
            l, j = sel
            if l >= len(net.sizes) or j >= net.sizes[l]:
                self.sel = None
        self.update()

    def _grid_image(self):
        G = 64
        xs = np.linspace(-1.1, 1.1, G)
        ys = np.linspace(1.1, -1.1, G)
        XX, YY = np.meshgrid(xs, ys)
        F = featurize(np.column_stack([XX.ravel(), YY.ravel()]), self.active)
        _, acts, *_ = self.net.forward(F)
        v = acts[self.sel[0]][:, self.sel[1]].reshape(G, G)
        out_layer = (self.sel[0] == self.net.L - 1)
        blue, orange = np.array(TH.heat_blue, float), np.array(TH.heat_orange, float)
        base = np.array(TH.heat_bg, float)
        if out_layer:
            t = np.clip(np.abs(v - 0.5) * 2, 0, 1)[..., None]
            fg = np.where((v > 0.5)[..., None], orange, blue)
        else:
            t = np.clip(np.abs(v), 0, 1)[..., None]
            fg = np.where((v > 0)[..., None], blue, orange)
        col = np.clip(base * (1 - t) + fg * t, 0, 255).astype(np.uint8)
        buf = np.ascontiguousarray(col)
        return QImage(buf.data, G, G, 3 * G, QImage.Format_RGB888).copy(), v

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        p.setFont(small_font(self))
        fm = p.fontMetrics()
        if self.net is None or self.sel is None:
            p.setPen(TH.muted)
            p.drawText(self.rect(), Qt.AlignCenter,
                       "click a neuron in the diagram to inspect it")
            return
        l, j = self.sel
        L = self.net.L
        key = (id(self.net), self.sel, self.epoch, round(self.net.delta_factor, 3),
               self.net.quantize_bias, self.net.ternary)
        if key != self._cache_key:
            self._cache_img = self._grid_image()
            self._cache_key = key
        img, vals = self._cache_img

        left = min(240.0, self.width() * 0.44)
        side = max(60.0, min(self.height() - 58.0, self.width() - left - 30.0))
        hx = left + 10 + (max(self.width() - left - 24, 70) - side) / 2
        rimg = QRectF(hx, 32, side, side)
        p.drawImage(rimg, img)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(TH.frame, 1))
        p.drawRect(rimg)

        name = "output · σ" if l == L - 1 else (f"h{l} · {self.net.activation}"
                                                if l > 0 else "input feature")
        p.setPen(TH.text)
        p.setFont(small_font(self, bold=True))
        p.drawText(QPointF(14, 20), f"{name} · unit {j}")
        p.setFont(small_font(self))

        y = 42.0
        if l == 0:
            lines = [("input feature — no incoming weights", TH.muted)]
        else:
            Wl = self.net.W[l - 1][:, j]
            ql = (self.net.quantized()[l - 1][:, j] if self.net.ternary else Wl)
            d = (self.net.deltas()[l - 1] if self.net.ternary else None)
            lines = []
            if d is not None:
                lines.append((f"Δ(l{l}) = ±{d:.3f}", C_RED))
            fz = "  [frozen]" if self.net.frozen[l - 1] else ""
            lines.append((f"bias {self.net.b[l - 1][j]:+.3f}{fz}", TH.muted))
            lines.append(("", TH.muted))
            lines.append(("incoming weights:", TH.text))
            for i in range(len(Wl)):
                prev_name = (self.in_labels[i] if l == 1 else f"h{l - 1}·{i}")
                w, q = float(Wl[i]), float(ql[i])
                sym = "+1" if q > 0 else ("−1" if q < 0 else "0")
                col = C_BLUE if q > 0 else (C_ORANGE if q < 0 else TH.muted)
                lines.append((f"  {prev_name:<7} {sym:>2}  ({w:+.2f})", col))
        for txt, col in lines[:int((self.height() - 48) / 14)]:
            p.setPen(col)
            p.drawText(QPointF(14, y), txt)
            y += 14

        if l > 0:
            a = np.asarray(vals)
            stats = f"plane mean|a| = {np.mean(np.abs(a)):.2f} · %a>0 = {100 * np.mean(a > 0):.0f}%"
            p.setPen(TH.muted)
            p.drawText(QPointF(hx, rimg.bottom() + 16), stats)
        p.setPen(TH.muted)
        p.drawText(QPointF(hx, 24), "activation over input plane (blue > 0 · orange < 0)")


class ConfusionCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.counts = None
        self.thr = 0.5

    def set_counts(self, tp, fp, fn, tn, thr=0.5):
        self.counts = None if tp is None else (tp, fp, fn, tn)
        self.thr = thr
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        p.setFont(small_font(self))
        fm = p.fontMetrics()
        if self.counts is None:
            p.setPen(TH.muted)
            p.drawText(self.rect(), Qt.AlignCenter, "no test data")
            return
        tp, fp, fn, tn = self.counts
        total = tp + fp + fn + tn or 1
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec > 0) else None

        top = 34.0
        s = max(24.0, min(self.height() - top - 44.0,
                          (self.width() - 290.0) / 2.0))
        mx, my = 48.0, top + 4.0
        cells = [(tp, 0, 0, True), (fp, 0, 1, False),
                 (fn, 1, 0, False), (tn, 1, 1, True)]
        mx_all = max(max(tp, fp, fn, tn), 1)
        p.setPen(TH.muted)
        p.drawText(QPointF(mx + s * 0.5 - fm.horizontalAdvance("pred 1") / 2, my - 6), "pred 1")
        p.drawText(QPointF(mx + s * 1.5 - fm.horizontalAdvance("pred 0") / 2, my - 6), "pred 0")
        p.drawText(QPointF(12, my + s * 0.5 + 4), "act 1")
        p.drawText(QPointF(12, my + s * 1.5 + 4), "act 0")
        bf = small_font(self, bold=True)
        for count, row, colidx, correct in cells:
            r = QRectF(mx + colidx * s, my + row * s, s - 2, s - 2)
            col = QColor(C_BLUE if correct else C_ORANGE)
            col.setAlpha(int(50 + 170 * count / mx_all))
            p.fillRect(r, col)
            p.setPen(QPen(TH.frame, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            p.setFont(bf)
            p.setPen(TH.text)
            p.drawText(r, Qt.AlignCenter, str(count))
            p.setFont(small_font(self))
            p.setPen(TH.muted)
            p.drawText(QPointF(r.center().x() - 14, r.bottom() - 3),
                       f"{100 * count / total:.0f}%")
        sx = mx + 2 * s + 24
        sy = my + 4
        lines = [("threshold", f"{self.thr:.2f}"),
                 ("accuracy", f"{acc * 100:.1f}%"),
                 ("precision (cl 1)", f"{prec:.3f}" if prec is not None else "—"),
                 ("recall (cl 1)", f"{rec:.3f}" if rec is not None else "—"),
                 ("F1 (cl 1)", f"{f1:.3f}" if f1 is not None else "—"),
                 ("support", f"{total}")]
        for name, val in lines:
            p.setPen(TH.muted)
            p.drawText(QPointF(sx, sy), name)
            p.setPen(TH.text)
            p.setFont(small_font(self, bold=True))
            p.drawText(QPointF(sx + 118, sy), val)
            p.setFont(small_font(self))
            sy += 16
        p.setPen(TH.muted)
        p.drawText(QPointF(14, self.height() - 8),
                   "threshold ⭐ snaps to best F1 — tune it like a real deployment knob")


class RocCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.scores = self.labels = None
        self.thr = 0.5

    def set_data(self, scores, labels, thr):
        self.scores, self.labels, self.thr = scores, labels, thr
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        p.setFont(small_font(self))
        if self.scores is None or len(self.scores) < 2:
            p.setPen(TH.muted)
            p.drawText(self.rect(), Qt.AlignCenter, "no test data")
            return
        pos = self.labels > 0.5
        n1, n0 = int(np.sum(pos)), int(np.sum(~pos))
        if n1 == 0 or n0 == 0:
            p.setPen(TH.muted)
            p.drawText(self.rect(), Qt.AlignCenter,
                       "ROC needs both classes in the test set")
            return
        s = np.asarray(self.scores, float)
        ts = np.linspace(1.0, 0.0, 101)
        tpr, fpr = [], []
        for t in ts:
            pred = s >= t
            tpr.append(np.sum(pred & pos) / n1)
            fpr.append(np.sum(pred & ~pos) / n0)
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0

        area = QRectF(44, 18, min(self.width() - 130, self.height() - 40),
                      self.height() - 40)
        X = lambda f: area.left() + f * area.width()
        Y = lambda t: area.bottom() - t * area.height()
        p.setPen(QPen(TH.grid, 1))
        for g in (0.25, 0.5, 0.75):
            p.drawLine(QPointF(X(g), area.top()), QPointF(X(g), area.bottom()))
            p.drawLine(QPointF(area.left(), Y(g)), QPointF(area.right(), Y(g)))
        p.setPen(QPen(TH.muted, 1, Qt.DashLine))
        p.drawLine(QPointF(X(0), Y(0)), QPointF(X(1), Y(1)))
        p.setPen(QPen(C_BLUE, 1.8))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF([QPointF(X(f), Y(t)) for f, t in zip(fpr, tpr)]))
        pred = s >= self.thr
        mp = QPointF(X(np.sum(pred & ~pos) / n0), Y(np.sum(pred & pos) / n1))
        p.setBrush(C_ORANGE)
        p.setPen(Qt.NoPen)
        p.drawEllipse(mp, 5, 5)
        p.setPen(TH.text)
        p.drawText(QPointF(area.right() + 8, area.top() + 12), f"AUC {auc:.3f}")
        p.setPen(TH.muted)
        p.drawText(QPointF(area.right() + 8, area.top() + 26), f"thr {self.thr:.2f}")
        p.drawText(QPointF(area.right() + 8, area.top() + 40), "orange =")
        p.drawText(QPointF(area.right() + 8, area.top() + 52), "your thr")
        p.drawText(QPointF(area.left(), self.height() - 4), "FPR →")
        p.drawText(QPointF(6, area.top() + 8), "TPR ↑")


class GradCanvas(QWidget):
    PALETTE = ["#1a73e8", "#e8710a", "#188038", "#d93025", "#9334e6", "#12a4af"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.hist = []

    def set_data(self, hist):
        self.hist = hist
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        p.setFont(small_font(self))
        fm = p.fontMetrics()
        p.setPen(TH.muted)
        p.drawText(QPointF(12, 16), "log₁₀ ‖∇W‖ per layer (lower = calmer training)")
        if len(self.hist) < 2 or not self.hist[0]:
            p.drawText(self.rect(), Qt.AlignCenter, "train to see gradient norms")
            return
        L = len(self.hist[0])
        logv = [[math.log10(max(row[l], 1e-8)) for row in self.hist] for l in range(L)]
        lo = min(min(s) for s in logv)
        hi = max(max(s) for s in logv)
        if hi - lo < 0.4:
            hi = lo + 0.4
        pad = 0.1 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        area = QRectF(44, 26, self.width() - 54, self.height() - 44)
        n = len(self.hist)
        ym = lambda v: area.bottom() - (v - lo) / (hi - lo) * area.height()
        for e in range(int(math.ceil(lo)), int(math.floor(hi)) + 1):
            p.setPen(QPen(TH.grid, 1))
            p.drawLine(QPointF(area.left(), ym(e)), QPointF(area.right(), ym(e)))
            p.setPen(TH.muted)
            p.drawText(QPointF(6, ym(e) + 4), f"1e{e}")
        xm = lambda i: area.left() + area.width() * i / (n - 1)
        lx = area.left() + 6
        for l in range(L):
            col = QColor(self.PALETTE[l % len(self.PALETTE)])
            p.setPen(QPen(col, 1.7))
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(xm(i), ym(v))
                                      for i, v in enumerate(logv[l])]))
            p.setPen(col)
            p.drawLine(QPointF(lx, 14), QPointF(lx + 10, 14))
            p.drawText(QPointF(lx + 14, 18), f"L{l + 1}")
            lx += 14 + fm.horizontalAdvance(f"L{l + 1}") + 10


# ------------------------------------------------------- small canvases ----
class LRTestCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.res, self.best_i = [], -1

    def set_results(self, res, best_i):
        self.res, self.best_i = res, best_i
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        p.setFont(small_font(self))
        if not self.res:
            return
        xs = [math.log10(r[0]) for r in self.res]
        ys = [r[2] for r in self.res]
        x0, x1 = min(xs) - 0.15, max(xs) + 0.15
        y0, y1 = min(ys), max(ys)
        if y1 - y0 < 1e-9:
            y1 = y0 + 1e-9
        pad = 0.12 * (y1 - y0)
        y0, y1 = y0 - pad, y1 + pad
        area = QRectF(48, 16, self.width() - 62, self.height() - 48)
        X = lambda v: area.left() + (v - x0) / (x1 - x0) * area.width()
        Y = lambda v: area.bottom() - (v - y0) / (y1 - y0) * area.height()
        p.setPen(QPen(TH.grid, 1))
        for k in range(3):
            yy = area.top() + area.height() * k / 2
            p.drawLine(QPointF(area.left(), yy), QPointF(area.right(), yy))
        p.setPen(QPen(TH.muted, 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF([QPointF(X(a), Y(b)) for a, b in zip(xs, ys)]))
        for i, (lr, _mn, fl, _fa) in enumerate(self.res):
            pt = QPointF(X(xs[i]), Y(fl))
            best = (i == self.best_i)
            p.setBrush(C_GREEN if best else QColor("#1a73e8"))
            p.setPen(Qt.NoPen)
            p.drawEllipse(pt, 5.5 if best else 4.2, 5.5 if best else 4.2)
            p.setPen(TH.muted)
            p.drawText(QPointF(pt.x() - 12, area.bottom() + 14), f"{lr:g}")
        p.setPen(C_GREEN)
        p.drawText(QPointF(area.left() + 6, area.top() + 12),
                   f"best lr = {self.res[self.best_i][0]:g}   (lowest final test loss)")
        p.setPen(TH.muted)
        p.drawText(QPointF(area.left(), self.height() - 6),
                   "learning rate (log) — final test loss after 25 epochs")


class CVCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.accs, self.mean = [], None

    def set_results(self, accs, mean):
        self.accs, self.mean = accs, mean
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), TH.bg)
        p.setFont(small_font(self))
        if not self.accs:
            return
        area = QRectF(44, 18, self.width() - 58, self.height() - 44)
        Y = lambda v: area.bottom() - v * area.height()
        p.setPen(QPen(TH.grid, 1))
        for g in (0.25, 0.5, 0.75, 1.0):
            p.drawLine(QPointF(area.left(), Y(g)), QPointF(area.right(), Y(g)))
            p.setPen(TH.muted)
            p.drawText(QPointF(8, Y(g) + 4), f"{g * 100:.0f}%")
            p.setPen(QPen(TH.grid, 1))
        n = len(self.accs)
        bw = area.width() / n
        for i, a in enumerate(self.accs):
            r = QRectF(area.left() + i * bw + bw * 0.18, Y(a), bw * 0.64,
                       area.height() - area.height() * a)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#1a73e8"))
            p.drawRect(r)
            p.setPen(TH.muted)
            p.drawText(QPointF(area.left() + i * bw + bw * 0.3, area.bottom() + 14),
                       f"fold {i + 1}")
            p.setPen(TH.text)
            p.drawText(QPointF(area.left() + i * bw + bw * 0.24, Y(a) - 4),
                       f"{a * 100:.1f}")
        if self.mean is not None:
            p.setPen(QPen(C_GREEN, 1.6, Qt.DashLine))
            p.drawLine(QPointF(area.left(), Y(self.mean)),
                       QPointF(area.right(), Y(self.mean)))
            p.setPen(C_GREEN)
            p.drawText(QPointF(area.left() + 4, Y(self.mean) - 4),
                       f"mean {self.mean * 100:.1f}%")
        p.setPen(TH.muted)
        p.drawText(QPointF(area.left(), self.height() - 4),
                   "held-out accuracy per fold")


# ------------------------------------------------------------ dialogs ------
def show_html(parent, title, html):
    d = QDialog(parent)
    d.setWindowTitle(title)
    d.resize(680, 560)
    v = QVBoxLayout(d)
    tb = QTextBrowser()
    tb.setHtml(html)
    v.addWidget(tb)
    bb = QDialogButtonBox(QDialogButtonBox.Close)
    bb.rejected.connect(d.reject)
    v.addWidget(bb)
    d.exec()


class LRTestDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("Learning-rate range test")
        self.resize(560, 430)
        self.win = win
        self.best_lr = None
        v = QVBoxLayout(self)
        self.canvas = LRTestCanvas()
        v.addWidget(self.canvas, 1)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply best lr")
        close = QPushButton("Close"); close.setObjectName("ghost")
        row.addStretch(1); row.addWidget(self.apply_btn); row.addWidget(close)
        v.addLayout(row)
        self.apply_btn.clicked.connect(self._apply)
        close.clicked.connect(self.reject)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self._run()
        finally:
            QApplication.restoreOverrideCursor()

    def _run(self):
        w = self.win
        if w.Ftr is None or len(w.Ftr) == 0:
            self.summary.setText("no data")
            return
        sizes = [len(w.active_features)] + list(w.layer_editor.layers) + [1]
        ref = TernaryNet(sizes, w.act_combo.currentText(), ternary=True,
                         optimizer=w._opt_name(), delta_factor=w._delta(),
                         quantize_bias=w.bias_chk.isChecked())
        lr_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        bs_txt = w.batch_combo.currentText()
        bs = len(w.Ftr) if bs_txt == "All" else max(1, int(bs_txt))
        rng = np.random.default_rng(w.seed_spin.value())
        res = []
        for lr in lr_list:
            net = ref.clone()
            fin_l = fin_a = None
            for _ in range(25):
                idx = rng.permutation(len(w.Ftr))
                for s in range(0, len(idx), bs):
                    sel = idx[s:s + bs]
                    net.train_batch(w.Ftr[sel], w.Ytr[sel], lr)
                fin_l, fin_a = w._metrics(net, w.Fte, w.Yte)
            res.append((lr, fin_l, fin_a))
        best_i = min(range(len(res)), key=lambda i: res[i][1])
        self.res, self.best_i = res, best_i
        self.best_lr = res[best_i][0]
        self.canvas.set_results(res, best_i)
        self.summary.setText(" · ".join(
            f"<b style='color:{'#188038' if i == best_i else 'inherit'}'>"
            f"{lr:g}: {fl:.3f} ({fa * 100:.0f}%)</b>"
            for i, (lr, fl, fa) in enumerate(res)))

    def _apply(self):
        if self.best_lr is None:
            return
        self.win.lr_combo.setCurrentText(f"{self.best_lr:g}")
        self.win.flags["lr_applied"] = True
        self.accept()


class CVDialog(QDialog):
    """K-fold cross-validation on the combined dataset with the current config."""

    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("K-fold cross-validation")
        self.resize(560, 480)
        self.win = win
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Folds"))
        self.k_combo = QComboBox(); self.k_combo.addItems(["2", "3", "5", "10"])
        self.k_combo.setCurrentText("5")
        row.addWidget(self.k_combo)
        row.addWidget(QLabel("Epochs/fold"))
        self.ep_spin = QSpinBox(); self.ep_spin.setRange(10, 500)
        self.ep_spin.setValue(60)
        row.addWidget(self.ep_spin)
        run = QPushButton("Run")
        row.addStretch(1); row.addWidget(run)
        v.addLayout(row)
        self.canvas = CVCanvas()
        v.addWidget(self.canvas, 1)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        close = QPushButton("Close"); close.setObjectName("ghost")
        crow = QHBoxLayout(); crow.addStretch(1); crow.addWidget(close)
        v.addLayout(crow)
        run.clicked.connect(self._run)
        close.clicked.connect(self.reject)
        self._run()

    def _run(self):
        w = self.win
        if w.Xtr is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            K = int(self.k_combo.currentText())
            eps = self.ep_spin.value()
            Xall = np.vstack([w.Xtr, w.Xte])
            yall = np.vstack([w.Ytr, w.Yte])
            sizes = [len(w.active_features)] + list(w.layer_editor.layers) + [1]
            lr = float(w.lr_combo.currentText())
            l2 = float(w.l2_spin.value())
            bs_txt = w.batch_combo.currentText()
            rng = np.random.default_rng(w.seed_spin.value())
            perm = rng.permutation(len(Xall))
            folds = np.array_split(perm, K)
            accs, losses = [], []
            for k in range(K):
                val = folds[k]
                tr = np.concatenate([folds[j] for j in range(K) if j != k])
                net = TernaryNet(sizes, w.act_combo.currentText(), ternary=True,
                                 optimizer=w._opt_name(), delta_factor=w._delta(),
                                 quantize_bias=w.bias_chk.isChecked())
                Ftr = featurize(Xall[tr], w.active_features)
                Fva = featurize(Xall[val], w.active_features)
                bs = len(tr) if bs_txt == "All" else max(1, int(bs_txt))
                for _ in range(eps):
                    idx = rng.permutation(len(tr))
                    for s in range(0, len(idx), bs):
                        sel = idx[s:s + bs]
                        net.train_batch(Ftr[sel], yall[tr][sel], lr, l2)
                vl, va = w._metrics(net, Fva, yall[val])
                losses.append(vl); accs.append(va)
            mean_acc = float(np.mean(accs))
            self.canvas.set_results(accs, mean_acc)
            self.summary.setText(
                f"test accuracy <b>{mean_acc * 100:.1f}% ± {np.std(accs) * 100:.1f}</b>"
                f"  ·  test loss {np.mean(losses):.3f} ± {np.std(losses):.3f}"
                f"  <span style='color:{'#80868b' if not TH.dark else '#9aa0a6'}'>"
                f"(mean ± std over {K} folds — report this, not a single split)</span>")
            w.flags["cv_done"] = True
        finally:
            QApplication.restoreOverrideCursor()


class PTQDialog(QDialog):
    """Compare: FP teacher vs FP→quantised (PTQ) vs ternary-trained (QAT)."""

    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("PTQ vs QAT")
        self.resize(520, 340)
        self.win = win
        v = QVBoxLayout(self)
        tb = QTextBrowser()
        v.addWidget(tb, 1)
        close = QPushButton("Close"); close.setObjectName("ghost")
        crow = QHBoxLayout(); crow.addStretch(1); crow.addWidget(close)
        v.addLayout(crow)
        close.clicked.connect(self.reject)
        w = win
        if w.fp_net is None or not w.hist_fp["tl"]:
            tb.setHtml("<h3>Quantise-after-training needs a trained FP net.</h3>"
                       "<p>Enable <b>Train full-precision shadow</b>, train for a "
                       "while, then reopen this dialog.</p>")
            return
        fp = w.fp_net
        ptq = TernaryNet(fp.sizes, fp.activation, ternary=True,
                         optimizer=fp.optimizer, delta_factor=fp.delta_factor,
                         quantize_bias=fp.quantize_bias)
        ptq.W = [x.copy() for x in fp.W]
        ptq.b = [x.copy() for x in fp.b]
        rows = []
        for name, note, net in (
                ("Full-precision (teacher)", "no quantization", fp),
                ("PTQ — quantise after training", "FP weights → Δ rule", ptq),
                ("QAT — trained ternary (yours)", "STE from step 1", w.net)):
            l, a = w._metrics(net, w.Fte, w.Yte)
            q = net.quantized()
            tot = sum(x.size for x in q)
            z = sum(int(np.count_nonzero(x == 0.0)) for x in q)
            rows.append((name, note, l, a, z / tot))
        h = ("<h3>The industrial question: train-then-quantise, or train-with-quantise?</h3>"
             "<table cellpadding='5' cellspacing='0' border='1' style='border-collapse:collapse'>"
             "<tr><th align='left'>model</th><th>test loss</th><th>test acc</th><th>zeros</th></tr>")
        for name, note, l, a, zr in rows:
            h += (f"<tr><td>{name}<br><small>{note}</small></td>"
                  f"<td align='center'>{l:.3f}</td><td align='center'>{a * 100:.1f}%</td>"
                  f"<td align='center'>{zr * 100:.0f}%</td></tr>")
        h += "</table><p>Typically <b>QAT ≥ PTQ</b>: the ternary net co-adapts to its "
             "own quantization noise during training, while PTQ jolts a trained FP net "
             "all at once. Quantify the gap above — this is the experiment practitioners "
             "run before every deployment.</p>"
        tb.setHtml(h)
        w.flags["ptq_done"] = True


class LogDialog(QDialog):
    COLS = ["time", "dataset", "arch", "features", "opt", "lr", "l2", "batch",
            "Δ", "epochs", "train_loss", "test_loss", "test_acc", "test_f1",
            "nonzero_%", "params", "seed"]

    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("Experiment log")
        self.resize(760, 460)
        self.win = win
        v = QVBoxLayout(self)
        self.tb = QTextBrowser()
        v.addWidget(self.tb, 1)
        row = QHBoxLayout()
        b_csv = QPushButton("Save CSV…")
        b_clr = QPushButton("Clear"); b_clr.setObjectName("ghost")
        close = QPushButton("Close"); close.setObjectName("ghost")
        row.addWidget(b_csv); row.addWidget(b_clr)
        row.addStretch(1); row.addWidget(close)
        v.addLayout(row)
        b_csv.clicked.connect(self._save)
        b_clr.clicked.connect(self._clear)
        close.clicked.connect(self.reject)
        self.refresh()

    def refresh(self):
        exp = self.win.experiments
        if not exp:
            self.tb.setHtml("<p>No runs logged yet. Train something, then press "
                            "<b>📓 Log run</b>.</p>")
            return
        h = ("<table cellpadding='4' cellspacing='0' border='1' "
             "style='border-collapse:collapse;font-size:11px'><tr>")
        for c in self.COLS:
            h += f"<th>{c}</th>"
        h += "</tr>"
        for e in exp:
            h += "<tr>" + "".join(
                f"<td>{e.get(c, '') if e.get(c, '') is not None else ''}</td>"
                for c in self.COLS) + "</tr>"
        self.tb.setHtml(h + "</table>")

    def _save(self):
        if not self.win.experiments:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save experiment log",
                                              "tnn_experiments.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=self.COLS)
            wr.writeheader()
            for e in self.win.experiments:
                wr.writerow({c: e.get(c, "") for c in self.COLS})
        self.win.statusBar().showMessage(f"experiment log → {path}", 4000)

    def _clear(self):
        self.win.experiments = []
        self.refresh()


CHALLENGES = [
    ("95%+ test accuracy", "any dataset — train well, use early stopping"),
    ("Perfect XOR", "100% test accuracy on XOR"),
    ("Half the weights are 0", "≥50% zeros — raise Δ factor or L2"),
    ("Ternary ≤ FP shadow", "match or beat the full-precision net's test loss"),
    ("Hand-crafted Gaussians", "0 epochs: sculpt weights in the Matrix tab"),
    ("Imbalance F1 ≥ 0.90", "class-1 share ≤ 30% — use ⭐ threshold + features"),
    ("Early stopping captured", "a best epoch was rolled back to"),
    ("LR test applied", "applied a learning rate found by the range test"),
    ("PTQ vs QAT compared", "ran Tools → PTQ vs QAT"),
    ("K-fold CV run", "ran Tools → K-fold cross-validation"),
    ("Greedy layer-wise", "≥80% test acc with some layer frozen"),
    ("Firmware export", "exported C inference code"),
]


class ChallengesDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("🎯 Challenges")
        self.resize(520, 520)
        self.win = win
        v = QVBoxLayout(self)
        self.counter = QLabel("")
        self.counter.setStyleSheet("font-weight:700;")
        v.addWidget(self.counter)
        self.lbls = []
        for title, hint in CHALLENGES:
            lbl = QLabel(f"⬜ <b>{title}</b><br><small style='color:#80868b'>{hint}</small>")
            lbl.setWordWrap(True)
            v.addWidget(lbl)
            self.lbls.append(lbl)
        row = QHBoxLayout()
        re = QPushButton("Recheck")
        close = QPushButton("Close"); close.setObjectName("ghost")
        row.addWidget(re); row.addStretch(1); row.addWidget(close)
        v.addLayout(row)
        re.clicked.connect(self.recheck)
        close.clicked.connect(self.reject)
        self.recheck()

    def recheck(self):
        states = self.win.challenge_states()
        for i, ok in enumerate(states):
            if ok:
                self.win.challenges_done.add(i)
        for i, lbl in enumerate(self.lbls):
            mark = "✅" if i in self.win.challenges_done else "⬜"
            title, hint = CHALLENGES[i]
            col = "#188038" if i in self.win.challenges_done else "#80868b"
            lbl.setText(f"{mark} <b>{title}</b><br>"
                        f"<small style='color:{col}'>{hint}</small>")
        self.counter.setText(f"{len(self.win.challenges_done)} / {len(CHALLENGES)} completed")


CONCEPTS_HTML = """
<h2>Ternary Neural Networks — the essentials</h2>
<h3>Why ternary weights?</h3>
<p>A weight in {−1, 0, +1} needs only <b>2 bits</b> (vs 32 for float32) and a
"multiplication" by it is just an <b>addition, subtraction, or nothing</b>.
16 weights pack into one 32-bit word — that is why ternary nets are a real
deployment technique for microcontrollers and edge accelerators.</p>
<h3>How quantization works here</h3>
<p>After every update, each layer's threshold is
<b>Δ = factor · mean|W|</b>. Latent weights with |W| ≤ Δ become <b>0</b>
(pruned!), others round to their <b>sign</b>. Watch the ±Δ red lines in the
Histogram tab — bars inside them are zeros in the deployed model.</p>
<h3>Straight-through estimator (STE)</h3>
<p>The sign/zero function has ~zero gradient, so backprop would die at the
quantizer. STE cheats: forward uses Ŵ, backward pretends Ŵ = W. Noisy, but it
works — the noise shows as jitter on the ternary curve vs the dashed FP curve.</p>
<h3>QAT vs PTQ</h3>
<p><b>QAT</b> (quantization-aware training) trains with quantization in the
loop. <b>PTQ</b> (post-training quantization) quantizes an already-trained FP
net in one shot. Tools → PTQ vs QAT runs both for you: QAT almost always wins,
because the network co-adapts to its own noise.</p>
<h3>Thresholds &amp; ROC</h3>
<p>0.5 is not sacred. Deployments move the decision threshold to trade precision
for recall. The <b>thr</b> slider updates the confusion matrix everywhere; the
⭐ button snaps to the F1-optimal threshold; the ROC tab shows the full trade-off
and its <b>AUC</b> (threshold-independent quality).</p>
<h3>Freezing &amp; greedy layer-wise training</h3>
<p>Frozen layers keep their weights while others train — the idea behind
transfer learning and old-school greedy pretraining. Freeze all but one layer,
train, then unlock the next.</p>
<h3>Reading the panels</h3>
<ul>
<li><b>Confusion</b> — with imbalanced data accuracy lies; watch F1.</li>
<li><b>ROC</b> — threshold-independent ranking quality (AUC 1.0 = perfect).</li>
<li><b>‖∇W‖</b> — vanishing (→1e-6) vs exploding (&gt;1e0) gradients.</li>
<li><b>Matrix</b> — click cells to hand-edit ternary weights (auto-pauses).</li>
<li><b>🎯 pick</b> — trace one sample's activations layer by layer.</li>
</ul>
<h3>Practical workflow</h3>
<ol>
<li>Simplest architecture that could work (× removes layers).</li>
<li>⚡ LR test → apply the best learning rate.</li>
<li>Train with early stopping on; enable jitter augmentation if train ≫ test.</li>
<li>Check ‖∇W‖; switch tanh↔ReLU if a layer's gradients die.</li>
<li>Log the run (📓) — compare configs in the experiment log, not in your head.</li>
<li>Confirm with K-fold CV; compare against the FP shadow; export C.</li>
</ol>
<p><b>Shortcuts:</b> Space = play/pause · → = +1 epoch · Ctrl+R = new data ·
Ctrl+D = dark theme · Ctrl+S/O/E = save/load/export · F1 = this page.</p>
"""

EXERCISES_HTML = """
<h2>Guided exercises</h2>
<ol>
<li><b>Warm-up.</b> Solve <i>Gaussians</i> with zero hidden layers (× the layers).
Note ternary logits are just a ±1/0 vote over features.</li>
<li><b>XOR minimal.</b> 100% test accuracy with ≤ 2 layers × 3 units; watch grey
(zero) edges appear.</li>
<li><b>Δ ablation.</b> Train <i>Circle</i> at Δ factor 0.1 then 0.9; read the
memory saving in the Deployment panel for each.</li>
<li><b>Price of ternary.</b> FP shadow on <i>Spiral</i> with only x₁,x₂; quantify
the gap, then close it with <i>sin x₁</i>.</li>
<li><b>LR finder.</b> ⚡ LR test → apply → confirm smoother loss. Then train at
lr = 1.0 and watch ‖∇W‖ explode + weights pile at ±1 (clipping).</li>
<li><b>Vanishing gradients.</b> 3 tanh layers at lr 0.001 → deep curves die in
‖∇W‖. Switch to ReLU, compare.</li>
<li><b>Imbalance.</b> Class-1 share 15% on <i>Gaussians</i>: hit 90% accuracy
while recall collapses; then push <b>F1</b> up (features, ⭐ threshold, L2).</li>
<li><b>Threshold surgery.</b> On imbalanced data, sweep the thr slider and watch
precision/recall trade places; press ⭐ and compare to the ROC's knee.</li>
<li><b>QAT vs PTQ.</b> Train the FP shadow to convergence, run Tools → PTQ vs
QAT, and report the accuracy gap. Repeat with a higher Δ factor — does PTQ
suffer more?</li>
<li><b>Cross-validation.</b> Run K-fold CV twice: tanh vs ReLU, same seed. Which
wins on mean ± std? A single split can lie; folds don't.</li>
<li><b>Greedy layer-wise.</b> Freeze all layers except L1, train 100 epochs;
freeze L1, unfreeze L2, train; unlock everything, fine-tune. Compare with
training everything at once.</li>
<li><b>Hand-craft.</b> Reset, then edit Matrix cells until <i>Gaussians</i> is
solved with 0 epochs — you are doing quantized integer programming by hand.</li>
<li><b>Deploy.</b> Export C (Tools/File → Export C) and read the generated
forward pass: multiply-free, 2-bit weights, fixed buffers.</li>
</ol>
"""


# ------------------------------------------------------------------ QSS ----
def build_qss(dark):
    if dark:
        win, card, chip, border = "#101114", "#202124", "#2a2b2f", "#3c4043"
        text, muted, inp, hov, sel = "#e8eaed", "#9aa0a6", "#2a2b2f", "#3a3d42", "#3c4043"
    else:
        win, card, chip, border = "#eef0f4", "#ffffff", "#f8f9fa", "#e1e4e8"
        text, muted, inp, hov, sel = "#202124", "#80868b", "#ffffff", "#e8eaed", "#e8f0fe"
    return f"""
* {{ font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif; font-size: 12px; color: {text}; }}
QMainWindow, QWidget {{ background: {win}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QGroupBox {{ background: {card}; border: 1px solid {border}; border-radius: 10px;
            margin-top: 12px; padding: 9px 9px 8px 9px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
                   left: 10px; padding: 0 4px; color: {text}; font-weight: 600; }}
QLabel#hint {{ color: {muted}; font-size: 11px; }}
QLabel#stat {{ font-weight: 700; }}
QFrame#chip {{ background: {chip}; border: 1px solid {border}; border-radius: 8px; }}
QPushButton {{ background: #1a73e8; color: white; border: none; border-radius: 6px;
              padding: 6px 10px; font-weight: 600; }}
QPushButton:hover {{ background: #1765cc; }}
QPushButton:pressed {{ background: #174ea6; }}
QPushButton:disabled {{ background: {hov}; color: {muted}; }}
QPushButton#ghost {{ background: {chip}; color: {text}; border: 1px solid {border}; }}
QPushButton#ghost:hover {{ background: {hov}; }}
QPushButton#mini {{ background: {chip}; color: {text}; border: 1px solid {border};
                   border-radius: 4px; padding: 1px 2px; font-weight: 700; }}
QPushButton#mini:hover {{ background: {hov}; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background: {inp}; border: 1px solid {border};
                                      border-radius: 6px; padding: 3px 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #8ab4f8; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 13px; height: 13px; margin: -5px 0;
                             border-radius: 6px; background: #1a73e8; }}
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 6px; background: {card}; }}
QTabBar::tab {{ background: transparent; color: {muted}; padding: 5px 10px; }}
QTabBar::tab:selected {{ color: {text}; border-bottom: 2px solid #1a73e8; }}
QMenuBar {{ background: {card}; border-bottom: 1px solid {border}; }}
QMenuBar::item:selected {{ background: {sel}; }}
QMenu {{ background: {card}; border: 1px solid {border}; }}
QMenu::item:selected {{ background: {sel}; }}
QStatusBar {{ background: {card}; border-top: 1px solid {border}; color: {muted}; }}
QToolTip {{ background: {text}; color: {card}; border: none; padding: 4px; }}
"""


# ----------------------------------------------------------- main window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ternary NN Playground 4.0  ·  weights ∈ {−1, 0, +1}")
        self.setMinimumSize(1220, 740)
        self.resize(1520, 900)

        self.rng = np.random.default_rng(0)
        self.net = None
        self.fp_net = None
        self.Xtr = self.Ytr = self.Xte = self.Yte = None
        self.Ftr = self.Fte = None
        self.active_features = [0, 1]
        self.epoch = 0
        self.hist = {"tl": [], "vl": [], "ta": [], "va": []}
        self.hist_fp = {"tl": [], "vl": [], "ta": [], "va": []}
        self.hist_g = []
        self.sel_neuron = None
        self.sel_sample = None
        self.best_vl = float("inf")
        self.best_epoch = None
        self.best_snap = None
        self.bad = 0
        self.thr = 0.5
        self.experiments = []
        self.flags = {"lr_applied": False, "exported_c": False,
                      "ptq_done": False, "cv_done": False}
        self.challenges_done = set()
        self._guard = True
        self._loading = False

        self._build_ui()
        self._build_menus()
        self._connect()
        self._guard = False

        self.rebuild_net()
        self.regenerate()

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.on_tick)

        QShortcut(QKeySequence("Space"), self).activated.connect(self.toggle_play)
        QShortcut(QKeySequence("Right"), self).activated.connect(lambda: self.run_epochs(1))
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.regenerate)

    # ---------------------------------------------------------- UI build --
    def _stat(self, grid, name, row):
        grid.addWidget(QLabel(name + ":"), row, 0)
        val = QLabel("—")
        val.setObjectName("stat")
        grid.addWidget(val, row, 1)
        return val

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(10)

        scroll = QScrollArea()
        scroll.setFixedWidth(316)
        scroll.setWidgetResizable(True)
        left = QWidget()
        scroll.setWidget(left)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(9)
        root.addWidget(scroll)

        gb = QGroupBox("1 · Data")
        dg = QGridLayout(gb)
        dg.setVerticalSpacing(6)
        dg.addWidget(QLabel("Dataset"), 0, 0)
        self.data_combo = QComboBox()
        self.data_combo.addItems(list(DATASETS.keys()))
        dg.addWidget(self.data_combo, 0, 1)
        dg.addWidget(QLabel("Noise"), 1, 0)
        self.noise = QSlider(Qt.Horizontal)
        self.noise.setRange(0, 60); self.noise.setValue(10)
        self.noise_lbl = QLabel("10 %"); self.noise_lbl.setMinimumWidth(36)
        nh = QHBoxLayout(); nh.addWidget(self.noise); nh.addWidget(self.noise_lbl)
        dg.addLayout(nh, 1, 1)
        dg.addWidget(QLabel("Samples"), 2, 0)
        self.n_spin = QSpinBox(); self.n_spin.setRange(100, 1000)
        self.n_spin.setSingleStep(50); self.n_spin.setValue(300)
        dg.addWidget(self.n_spin, 2, 1)
        dg.addWidget(QLabel("Train split"), 3, 0)
        self.ratio = QSlider(Qt.Horizontal)
        self.ratio.setRange(20, 80); self.ratio.setValue(50)
        self.ratio_lbl = QLabel("50 %"); self.ratio_lbl.setMinimumWidth(36)
        rh = QHBoxLayout(); rh.addWidget(self.ratio); rh.addWidget(self.ratio_lbl)
        dg.addLayout(rh, 3, 1)
        dg.addWidget(QLabel("Class-1 share"))
        self.imb = QSlider(Qt.Horizontal)
        self.imb.setRange(10, 90); self.imb.setValue(50)
        self.imb_lbl = QLabel("50 %"); self.imb_lbl.setMinimumWidth(36)
        self.imb.setToolTip("Resample so class 1 is this share — imbalance makes\n"
                            "accuracy misleading; watch precision/recall diverge.")
        ih = QHBoxLayout(); ih.addWidget(self.imb); ih.addWidget(self.imb_lbl)
        dg.addLayout(ih, 4, 1)
        dg.addWidget(QLabel("Seed"), 5, 0)
        seed_row = QHBoxLayout()
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 99999)
        self.seed_btn = QPushButton("🎲"); self.seed_btn.setObjectName("ghost")
        self.seed_btn.setFixedWidth(34)
        seed_row.addWidget(self.seed_spin); seed_row.addWidget(self.seed_btn)
        dg.addLayout(seed_row, 5, 1)
        self.btn_data = QPushButton("↻ Regenerate data")
        self.btn_data.setObjectName("ghost")
        dg.addWidget(self.btn_data, 6, 0, 1, 2)
        lv.addWidget(gb)

        gb = QGroupBox("2 · Input features")
        fg = QGridLayout(gb)
        self.feat_checks = []
        for i, name in enumerate(FEATURES):
            cb = QCheckBox(name)
            cb.setChecked(i in (0, 1))
            self.feat_checks.append(cb)
            fg.addWidget(cb, i // 2, i % 2)
        tip = QLabel("Engineered features make the hard datasets learnable — "
                     "the ternary net pays for each one in memory.")
        tip.setObjectName("hint"); tip.setWordWrap(True)
        fg.addWidget(tip, 4, 0, 1, 2)
        lv.addWidget(gb)

        gb = QGroupBox("3 · Architecture & quantization")
        ag = QGridLayout(gb)
        ag.setVerticalSpacing(6)
        ag.addWidget(QLabel("Hidden layers  (− / + / ×)"), 0, 0)
        self.layer_editor = LayerEditor()
        ag.addWidget(self.layer_editor, 1, 0, 1, 2)
        ag.addWidget(QLabel("Activation"), 2, 0)
        self.act_combo = QComboBox(); self.act_combo.addItems(["tanh", "ReLU"])
        ag.addWidget(self.act_combo, 2, 1)
        ag.addWidget(QLabel("Δ factor"), 3, 0)
        self.delta_slider = QSlider(Qt.Horizontal)
        self.delta_slider.setRange(10, 95); self.delta_slider.setValue(70)
        self.delta_lbl = QLabel("0.70·mean|W|"); self.delta_lbl.setMinimumWidth(76)
        self.delta_slider.setToolTip("Quantisation threshold Δ = factor·mean|W|.\n"
                                     "Latent weights with |W| ≤ Δ become 0.")
        dh = QHBoxLayout(); dh.addWidget(self.delta_slider); dh.addWidget(self.delta_lbl)
        ag.addLayout(dh, 3, 1)
        self.bias_chk = QCheckBox("Ternary biases (quantized like weights)")
        ag.addWidget(self.bias_chk, 4, 0, 1, 2)
        ag.addWidget(QLabel("Freeze layer"), 5, 0)
        self.freeze_row = QHBoxLayout()
        self.freeze_row.setSpacing(6)
        ag.addLayout(self.freeze_row, 5, 1)
        lv.addWidget(gb)

        gb = QGroupBox("4 · Training")
        tg = QGridLayout(gb)
        tg.setVerticalSpacing(6)
        tg.addWidget(QLabel("Optimizer"), 0, 0)
        self.opt_combo = QComboBox(); self.opt_combo.addItems(["SGD+momentum", "Adam"])
        tg.addWidget(self.opt_combo, 0, 1)
        tg.addWidget(QLabel("Learning rate"), 1, 0)
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0"])
        self.lr_combo.setCurrentText("0.03")
        tg.addWidget(self.lr_combo, 1, 1)
        tg.addWidget(QLabel("LR decay"), 2, 0)
        self.decay_spin = QDoubleSpinBox()
        self.decay_spin.setRange(0.0, 10.0); self.decay_spin.setSingleStep(0.1)
        self.decay_spin.setDecimals(2); self.decay_spin.setValue(0.0)
        self.decay_spin.setSuffix(" %/ep")
        tg.addWidget(self.decay_spin, 2, 1)
        tg.addWidget(QLabel("L2 (weight decay)"), 3, 0)
        self.l2_spin = QDoubleSpinBox()
        self.l2_spin.setRange(0.0, 0.1); self.l2_spin.setSingleStep(0.001)
        self.l2_spin.setDecimals(3)
        tg.addWidget(self.l2_spin, 3, 1)
        tg.addWidget(QLabel("Batch size"), 4, 0)
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["1", "5", "10", "20", "50", "All"])
        self.batch_combo.setCurrentText("10")
        tg.addWidget(self.batch_combo, 4, 1)
        tg.addWidget(QLabel("Speed"), 5, 0)
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(1, 20); self.speed.setValue(2)
        self.speed_lbl = QLabel("2 ep/tick"); self.speed_lbl.setMinimumWidth(58)
        sh = QHBoxLayout(); sh.addWidget(self.speed); sh.addWidget(self.speed_lbl)
        tg.addLayout(sh, 5, 1)
        augr = QHBoxLayout()
        self.aug_chk = QCheckBox("jitter augment")
        self.aug_chk.setToolTip("Add gaussian noise to raw inputs each batch\n"
                                "(regularisation — closes the train/test gap).")
        self.aug_spin = QDoubleSpinBox()
        self.aug_spin.setRange(0.0, 0.25); self.aug_spin.setSingleStep(0.01)
        self.aug_spin.setValue(0.05); self.aug_spin.setDecimals(2)
        augr.addWidget(self.aug_chk); augr.addWidget(self.aug_spin); augr.addStretch(1)
        tg.addLayout(augr, 6, 0, 1, 2)
        self.es_chk = QCheckBox("Early stopping")
        self.es_chk.setToolTip("When test loss stops improving, roll back to the\n"
                               "best epoch and pause.")
        self.es_spin = QSpinBox()
        self.es_spin.setRange(5, 500); self.es_spin.setValue(25)
        self.es_spin.setSuffix(" ep"); self.es_spin.setEnabled(False)
        esr = QHBoxLayout()
        esr.addWidget(self.es_chk); esr.addWidget(self.es_spin)
        tg.addLayout(esr, 7, 0, 1, 2)
        self.fp_chk = QCheckBox("Train full-precision shadow (dashed = FP)")
        tg.addWidget(self.fp_chk, 8, 0, 1, 2)
        btns = QHBoxLayout()
        self.play_btn = QPushButton("▶  Train")
        self.step_btn = QPushButton("+1"); self.step_btn.setObjectName("ghost")
        self.step10_btn = QPushButton("+10"); self.step10_btn.setObjectName("ghost")
        self.reset_btn = QPushButton("Reset"); self.reset_btn.setObjectName("ghost")
        for b in (self.play_btn, self.step_btn, self.step10_btn, self.reset_btn):
            btns.addWidget(b)
        tg.addLayout(btns, 9, 0, 1, 2)
        btns2 = QHBoxLayout()
        self.lrtest_btn = QPushButton("⚡ LR test")
        self.lrtest_btn.setObjectName("ghost")
        self.btn_logrun = QPushButton("📓 Log run")
        self.btn_logrun.setObjectName("ghost")
        self.btn_logrun.setToolTip("Append this run's config + metrics to the experiment log.")
        btns2.addWidget(self.lrtest_btn); btns2.addWidget(self.btn_logrun)
        btns2.addStretch(1)
        tg.addLayout(btns2, 10, 0, 1, 2)
        read = QGridLayout()
        self.lbl_epoch_val = self._stat(read, "Epoch", 0)
        self.lbl_tl_val = self._stat(read, "Train loss", 1)
        self.lbl_vl_val = self._stat(read, "Test loss", 2)
        self.lbl_ta_val = self._stat(read, "Train acc", 3)
        self.lbl_va_val = self._stat(read, "Test acc", 4)
        tg.addLayout(read, 11, 0, 1, 2)
        lv.addWidget(gb)

        gb = QGroupBox("5 · Deployment cost (ternary vs FP32)")
        dv = QVBoxLayout(gb)
        self.d_mem = QLabel("—"); self.d_mem.setObjectName("hint")
        self.d_ops = QLabel("—"); self.d_ops.setObjectName("hint")
        self.d_ram = QLabel("—"); self.d_ram.setObjectName("hint")
        for wdg in (self.d_mem, self.d_ops, self.d_ram):
            wdg.setWordWrap(True)
            dv.addWidget(wdg)
        drow = QHBoxLayout()
        self.btn_c = QPushButton("⬇ Export C for MCU")
        self.btn_npz = QPushButton("⬇ .npz")
        self.btn_npz.setObjectName("ghost")
        drow.addWidget(self.btn_c); drow.addWidget(self.btn_npz)
        dv.addLayout(drow)
        lv.addWidget(gb)

        note = QLabel("Forward pass uses ternary Ŵ ∈ {−1, 0, +1}; gradients flow through "
                      "the quantiser (STE) into latent weights clipped to [−1, 1]. "
                      "Click neurons & matrix cells, pick samples — Tools menu has "
                      "CV, PTQ-vs-QAT, log and challenges.")
        note.setObjectName("hint"); note.setWordWrap(True)
        lv.addWidget(note)
        lv.addStretch(1)

        # -------- center: decision boundary
        gb_b = QGroupBox("Decision boundary · p(class 1 | x)")
        bv = QVBoxLayout(gb_b)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Weights:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Ternary Ŵ", "Full-precision W"])
        self.view_combo.setEnabled(False)
        hdr.addWidget(self.view_combo)
        self.chk_test = QCheckBox("test pts"); self.chk_test.setChecked(True)
        self.chk_disc = QCheckBox("discretize")
        hdr.addWidget(self.chk_test); hdr.addWidget(self.chk_disc)
        self.pick_btn = QPushButton("🎯 pick")
        self.pick_btn.setObjectName("ghost")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip("Click near a data point to trace it through the network.")
        self.clear_btn = QPushButton("✕")
        self.clear_btn.setObjectName("ghost")
        self.clear_btn.setFixedWidth(26)
        hdr.addWidget(self.pick_btn); hdr.addWidget(self.clear_btn)
        hdr.addStretch(1)
        self.btn_rv = QPushButton("⌂ view"); self.btn_rv.setObjectName("ghost")
        hdr.addWidget(self.btn_rv)
        bv.addLayout(hdr)
        self.boundary = BoundaryCanvas()
        bv.addWidget(self.boundary)
        root.addWidget(gb_b, 1)

        # -------- right: network / metrics / inspector tabs
        right = QSplitter(Qt.Vertical)
        right.setMinimumWidth(410)
        right.setMaximumWidth(560)
        right.setChildrenCollapsible(False)
        right.setHandleWidth(8)

        gb_n = QGroupBox("Network — blue +1 · orange −1 · grey 0 · dot = bias · click a neuron")
        nv = QVBoxLayout(gb_n)
        self.netview = NetworkCanvas()
        nv.addWidget(self.netview)
        self.trace_lbl = QLabel("no sample selected — 🎯 pick one on the boundary plot")
        self.trace_lbl.setObjectName("hint")
        self.trace_lbl.setWordWrap(True)
        nv.addWidget(self.trace_lbl)
        right.addWidget(gb_n)

        gb_l = QGroupBox("Metrics")
        lv2 = QVBoxLayout(gb_l)
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Curve:"))
        self.loss_mode = QComboBox(); self.loss_mode.addItems(["Loss", "Accuracy"])
        lrow.addWidget(self.loss_mode)
        lrow.addStretch(1)
        lrow.addWidget(QLabel("thr"))
        self.thr_slider = QSlider(Qt.Horizontal)
        self.thr_slider.setRange(5, 95); self.thr_slider.setValue(50)
        self.thr_slider.setFixedWidth(80)
        self.thr_slider.setToolTip("Decision threshold — updates confusion, ROC marker,\n"
                                   "sample verdicts and discretize view.")
        self.thr_lbl = QLabel("0.50")
        self.thr_btn = QPushButton("⭐F1")
        self.thr_btn.setObjectName("ghost")
        self.thr_btn.setToolTip("Snap the threshold to the F1-optimal value on the test set.")
        lrow.addWidget(self.thr_slider); lrow.addWidget(self.thr_lbl)
        lrow.addWidget(self.thr_btn)
        self.lbl_best = QLabel("best: —")
        self.lbl_best.setObjectName("hint")
        self.btn_best = QPushButton("⤺ best")
        self.btn_best.setObjectName("ghost")
        self.btn_best.setEnabled(False)
        lrow.addWidget(self.lbl_best); lrow.addWidget(self.btn_best)
        lv2.addLayout(lrow)
        self.lossview = LossCanvas()
        lv2.addWidget(self.lossview)
        right.addWidget(gb_l)

        gb_w = QGroupBox("Inspector")
        wv = QVBoxLayout(gb_w)
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Layer:"))
        self.hist_combo = QComboBox()
        wrow.addWidget(self.hist_combo, 1)
        wv.addLayout(wrow)
        self.wtabs = QTabWidget()
        self.histview = HistCanvas()
        self.matrixview = MatrixCanvas()
        self.neuronview = NeuronCanvas()
        self.confview = ConfusionCanvas()
        self.rocview = RocCanvas()
        self.gradview = GradCanvas()
        self.wtabs.addTab(self.histview, "Histogram")
        self.wtabs.addTab(self.matrixview, "Matrix ✎")
        self.wtabs.addTab(self.neuronview, "Neuron")
        self.wtabs.addTab(self.confview, "Confusion")
        self.wtabs.addTab(self.rocview, "ROC")
        self.wtabs.addTab(self.gradview, "‖∇W‖")
        wv.addWidget(self.wtabs)
        right.addWidget(gb_w)
        right.setSizes([330, 190, 300])
        root.addWidget(right)

        self.statusBar().showMessage(
            "▶ Train (Space) · → = +1 epoch · Ctrl+R data · click neurons / matrix cells · "
            "🎯 pick samples · Tools: CV, PTQ-vs-QAT, log, challenges")

    def _build_menus(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        a = m_file.addAction("Save model…"); a.setShortcut(QKeySequence("Ctrl+S"))
        a.triggered.connect(self._save_model)
        a = m_file.addAction("Load model…"); a.setShortcut(QKeySequence("Ctrl+O"))
        a.triggered.connect(self._load_model)
        m_file.addSeparator()
        a = m_file.addAction("Export ternary weights…"); a.setShortcut(QKeySequence("Ctrl+E"))
        a.triggered.connect(self._export_ternary)
        a = m_file.addAction("Export C for MCU…")
        a.triggered.connect(self._export_c)
        a = m_file.addAction("Export boundary PNG…")
        a.triggered.connect(self._export_png)
        m_file.addSeparator()
        a = m_file.addAction("Quit"); a.setShortcut(QKeySequence("Ctrl+Q"))
        a.triggered.connect(self.close)

        m_tools = mb.addMenu("&Tools")
        m_tools.addAction("⚡ LR range test…").triggered.connect(
            lambda: LRTestDialog(self).exec())
        m_tools.addAction("K-fold cross-validation…").triggered.connect(
            lambda: CVDialog(self).exec())
        m_tools.addAction("PTQ vs QAT…").triggered.connect(
            lambda: PTQDialog(self).exec())
        m_tools.addAction("Experiment log…").triggered.connect(
            lambda: LogDialog(self).exec())
        m_tools.addSeparator()
        m_tools.addAction("🎯 Challenges…").triggered.connect(
            lambda: ChallengesDialog(self).exec())

        m_view = mb.addMenu("&View")
        self.act_dark = m_view.addAction("Dark theme")
        self.act_dark.setCheckable(True)
        self.act_dark.setShortcut(QKeySequence("Ctrl+D"))
        self.act_dark.toggled.connect(self.set_theme)
        m_view.addAction("Reset boundary view").triggered.connect(self.boundary.reset_view)

        m_help = mb.addMenu("&Help")
        a = m_help.addAction("Concepts & tips"); a.setShortcut(QKeySequence("F1"))
        a.triggered.connect(lambda: show_html(self, "Concepts & tips", CONCEPTS_HTML))
        m_help.addAction("Guided exercises").triggered.connect(
            lambda: show_html(self, "Guided exercises", EXERCISES_HTML))
        m_help.addAction("About").triggered.connect(self._about)

    def _connect(self):
        self.data_combo.currentIndexChanged.connect(self.regenerate)
        self.noise.valueChanged.connect(self._on_noise)
        self.n_spin.valueChanged.connect(self.regenerate)
        self.ratio.valueChanged.connect(self._on_ratio)
        self.imb.valueChanged.connect(self._on_imb)
        self.seed_spin.valueChanged.connect(self._on_seed)
        self.seed_btn.clicked.connect(
            lambda: self.seed_spin.setValue(random.randint(0, 99999)))
        self.btn_data.clicked.connect(self.regenerate)
        for cb in self.feat_checks:
            cb.toggled.connect(self.on_feature_toggled)
        self.layer_editor.changed.connect(self.rebuild_net)
        self.act_combo.currentTextChanged.connect(self.rebuild_net)
        self.opt_combo.currentTextChanged.connect(self.rebuild_net)
        self.delta_slider.valueChanged.connect(self._on_delta)
        self.bias_chk.toggled.connect(self._on_bias)
        self.speed.valueChanged.connect(lambda v: self.speed_lbl.setText(f"{v} ep/tick"))
        self.aug_chk.toggled.connect(lambda *_: None)
        self.es_chk.toggled.connect(self.es_spin.setEnabled)
        self.fp_chk.toggled.connect(self.on_fp_toggled)
        self.play_btn.clicked.connect(self.toggle_play)
        self.step_btn.clicked.connect(lambda: self.run_epochs(1))
        self.step10_btn.clicked.connect(lambda: self.run_epochs(10))
        self.reset_btn.clicked.connect(self.rebuild_net)
        self.lrtest_btn.clicked.connect(lambda: LRTestDialog(self).exec())
        self.btn_logrun.clicked.connect(self._log_run)
        self.thr_slider.valueChanged.connect(self._on_thr)
        self.thr_btn.clicked.connect(self._snap_f1)
        self.btn_best.clicked.connect(self._restore_best_clicked)
        self.hist_combo.currentIndexChanged.connect(self.update_hist_canvas)
        self.view_combo.currentIndexChanged.connect(lambda _: self.push_all())
        self.chk_test.toggled.connect(self._on_view_flags)
        self.chk_disc.toggled.connect(self._on_view_flags)
        self.loss_mode.currentIndexChanged.connect(lambda _: self.push_all())
        self.btn_rv.clicked.connect(self.boundary.reset_view)
        self.pick_btn.toggled.connect(self.boundary.set_pick_mode)
        self.clear_btn.clicked.connect(self._clear_sample)
        self.boundary.samplePicked.connect(self._on_sample_picked)
        self.netview.neuronClicked.connect(self._on_neuron_clicked)
        self.matrixview.cellClicked.connect(self._on_matrix_cell)
        self.btn_c.clicked.connect(self._export_c)
        self.btn_npz.clicked.connect(self._export_ternary)
        for w in self.findChildren((QPushButton, QComboBox, QCheckBox)):
            w.setFocusPolicy(Qt.NoFocus)

    # ------------------------------------------------------------- slots --
    def _on_noise(self, v):
        self.noise_lbl.setText(f"{v} %")
        if not self._loading:
            self.regenerate()

    def _on_ratio(self, v):
        self.ratio_lbl.setText(f"{v} %")
        if not self._loading:
            self.regenerate()

    def _on_imb(self, v):
        self.imb_lbl.setText(f"{v} %")
        if not self._loading:
            self.regenerate()

    def _on_seed(self):
        if self._loading:
            return
        self.rng = np.random.default_rng(self.seed_spin.value())
        self.rebuild_net()
        self.regenerate()

    def _on_delta(self, v):
        self.delta_lbl.setText(f"{v / 100:.2f}·mean|W|")
        if self._loading:
            return
        for net in (self.net, self.fp_net):
            if net is not None:
                net.delta_factor = v / 100
        self.push_all()

    def _on_bias(self, on):
        if self._loading:
            return
        for net in (self.net, self.fp_net):
            if net is not None:
                net.quantize_bias = bool(on)
        self.push_all()

    def _on_thr(self, v):
        self.thr_lbl.setText(f"{v / 100:.2f}")
        self.thr = v / 100
        if not self._loading:
            self.push_all()

    def _on_view_flags(self, *_):
        self.boundary.set_flags(self.chk_test.isChecked(), self.chk_disc.isChecked())

    def on_feature_toggled(self, *_):
        if self._guard or self._loading:
            return
        checked = [i for i, c in enumerate(self.feat_checks) if c.isChecked()]
        if not checked:
            self._guard = True
            self.sender().setChecked(True)
            self._guard = False
            return
        self.active_features = checked
        self.rebuild_net()

    def on_fp_toggled(self, *_):
        if self._loading:
            return
        if self.fp_chk.isChecked() and self.net is not None:
            self.fp_net = TernaryNet(self.net.sizes, self.act_combo.currentText(),
                                     ternary=False, optimizer=self._opt_name(),
                                     delta_factor=self._delta(),
                                     quantize_bias=self.bias_chk.isChecked())
            self.fp_net.frozen = list(self.net.frozen)
            self.hist_fp = {"tl": [], "vl": [], "ta": [], "va": []}
        else:
            self.fp_net = None
            self.hist_fp = {"tl": [], "vl": [], "ta": [], "va": []}
            self.view_combo.setCurrentIndex(0)
        self.view_combo.setEnabled(self.fp_net is not None)
        self.push_all()

    def _on_freeze(self, k, on):
        for net in (self.net, self.fp_net):
            if net is not None:
                net.frozen[k] = bool(on)
        if on:
            self.statusBar().showMessage(
                f"layer {k + 1} frozen — greedy layer-wise training (unlock others "
                f"later, or fine-tune everything at the end)", 5000)

    def _refresh_freeze_row(self):
        while self.freeze_row.count():
            it = self.freeze_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        L = self.net.L if self.net else 0
        for l in range(L):
            cb = QCheckBox(f"L{l + 1}")
            cb.setFocusPolicy(Qt.NoFocus)
            cb.setToolTip("freeze this weight layer (greedy layer-wise training)")
            cb.toggled.connect(lambda on, k=l: self._on_freeze(k, on))
            self.freeze_row.addWidget(cb)
        self.freeze_row.addStretch(1)

    def _on_sample_picked(self, d):
        self.sel_sample = d
        self.boundary.set_selected((d["x"], d["y"]) if d else None)
        self.push_all()

    def _clear_sample(self):
        self.sel_sample = None
        self.boundary.set_selected(None)
        self.push_all()

    def _on_neuron_clicked(self, l, j):
        self.sel_neuron = (l, j)
        self.wtabs.setCurrentIndex(2)
        self.push_all()

    def _on_matrix_cell(self, l, i, j):
        if self.net is None or not self.net.ternary:
            return
        if self.timer.isActive():
            self.toggle_play()
        q = self.net.quantized()[l][i, j]
        if q > 0:
            new_lat, sym = 0.0, "0"
        elif q < 0:
            new_lat, sym = 0.9, "+1"
        else:
            new_lat, sym = -0.9, "−1"
        self.net.W[l][i, j] = new_lat
        self.push_all()
        self.statusBar().showMessage(
            f"hand-edited L{l + 1}.w[{i},{j}] → {sym}  (training auto-paused — "
            f"resume or reset to retrain)", 5000)

    def set_theme(self, dark):
        TH.dark = bool(dark)
        TH.refresh()
        QApplication.instance().setStyleSheet(build_qss(TH.dark))
        for c in (self.boundary, self.netview, self.lossview, self.histview,
                  self.matrixview, self.neuronview, self.confview, self.rocview,
                  self.gradview):
            c.update()

    def _about(self):
        QMessageBox.about(
            self, "About",
            "<b>Ternary NN Playground 4.0</b><br><br>"
            "Forward passes multiply with weights quantised to <b>{−1, 0, +1}</b>; "
            "training uses the straight-through estimator. Practice tools: threshold "
            "tuning + ROC/AUC, K-fold CV, PTQ-vs-QAT, experiment log, augmentation, "
            "layer freezing, hand-editable weights, challenges, and MCU C export.")

    # --------------------------------------------------------- app logic --
    def _opt_name(self):
        return "adam" if self.opt_combo.currentText() == "Adam" else "sgd"

    def _delta(self):
        return self.delta_slider.value() / 100

    def _common(self):
        return dict(optimizer=self._opt_name(), delta_factor=self._delta(),
                    quantize_bias=self.bias_chk.isChecked())

    def _reset_progress(self):
        self.epoch = 0
        self.hist = {"tl": [], "vl": [], "ta": [], "va": []}
        self.hist_fp = {"tl": [], "vl": [], "ta": [], "va": []}
        self.hist_g = []
        self.best_vl = float("inf")
        self.best_epoch = None
        self.best_snap = None
        self.bad = 0

    def regenerate(self, *_):
        if self._loading:
            return
        n = self.n_spin.value()
        sigma = self.noise.value() / 100 * 0.4
        pool = min(max(n * 3, 240), 3000)
        X, y = generate_data(self.data_combo.currentText(), pool, sigma, self.rng)
        X, y = subsample_imbalance(X, y, n, self.imb.value() / 100, self.rng)
        n_tr = max(4, int(round(len(X) * self.ratio.value() / 100)))
        self.Xtr, self.Ytr = X[:n_tr], y[:n_tr]
        self.Xte, self.Yte = X[n_tr:], y[n_tr:]
        self.Ftr = featurize(self.Xtr, self.active_features)
        self.Fte = featurize(self.Xte, self.active_features)
        self._reset_progress()
        self.sel_sample = None
        self.boundary.set_selected(None)
        self.push_all()

    def rebuild_net(self, *_):
        if self._loading:
            return
        hidden = list(self.layer_editor.layers) or [4]
        sizes = [len(self.active_features)] + hidden + [1]
        self.net = TernaryNet(sizes, self.act_combo.currentText(), ternary=True,
                              **self._common())
        self.fp_net = (TernaryNet(sizes, self.act_combo.currentText(),
                                  ternary=False, **self._common())
                       if self.fp_chk.isChecked() else None)
        self._reset_progress()
        self.sel_neuron = None
        self.sel_sample = None
        self.boundary.set_selected(None)
        self._refresh_freeze_row()
        self.hist_combo.blockSignals(True)
        self.hist_combo.clear()
        self.hist_combo.addItem("All layers")
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.hist_combo.addItem(f"Layer {i + 1}   ({a} → {b})")
        self.hist_combo.blockSignals(False)
        self.hist_combo.setCurrentIndex(0)
        self.view_combo.setEnabled(self.fp_net is not None)
        if self.fp_net is None:
            self.view_combo.setCurrentIndex(0)
        self.push_all()

    def _restore_best(self):
        if not self.best_snap:
            return
        self.net.restore(self.best_snap["net"])
        if self.fp_net is not None and self.best_snap.get("fp"):
            self.fp_net.restore(self.best_snap["fp"])
        self.bad = 0

    def _restore_best_clicked(self):
        if not self.best_snap:
            return
        self._restore_best()
        self.push_all()
        self.statusBar().showMessage(
            f"restored best snapshot (epoch {self.best_epoch}, "
            f"test loss {self.best_vl:.3f})", 4000)

    def run_epochs(self, k):
        if self.net is None or self.Ftr is None or len(self.Ftr) == 0:
            return
        lr0 = float(self.lr_combo.currentText())
        l2 = float(self.l2_spin.value())
        btxt = self.batch_combo.currentText()
        bs = len(self.Ftr) if btxt == "All" else max(1, int(btxt))
        patience = self.es_spin.value() if self.es_chk.isChecked() else 0
        aug = float(self.aug_spin.value()) if self.aug_chk.isChecked() else 0.0
        for _ in range(k):
            lr = lr0 * (1.0 - self.decay_spin.value() / 100.0) ** self.epoch
            idx = self.rng.permutation(len(self.Ftr))
            gsum = [0.0] * self.net.L
            nb = 0
            for s in range(0, len(idx), bs):
                sel = idx[s:s + bs]
                if aug > 0:                       # jitter in raw input space
                    Xb = self.Xtr[sel] + self.rng.normal(0.0, aug, (len(sel), 2))
                    Fb = featurize(Xb, self.active_features)
                else:
                    Fb = self.Ftr[sel]
                self.net.train_batch(Fb, self.Ytr[sel], lr, l2)
                gn = self.net.last_gnorms
                if len(gn) == self.net.L:
                    for i in range(self.net.L):
                        gsum[i] += gn[i]
                if self.fp_net is not None:
                    self.fp_net.train_batch(Fb, self.Ytr[sel], lr, l2)
                nb += 1
            self.epoch += 1
            tl, ta = self._metrics(self.net, self.Ftr, self.Ytr)
            vl, va = self._metrics(self.net, self.Fte, self.Yte)
            self.hist["tl"].append(tl); self.hist["ta"].append(ta)
            self.hist["vl"].append(vl); self.hist["va"].append(va)
            self.hist_g.append([g / max(nb, 1) for g in gsum])
            if self.fp_net is not None:
                fl, fa = self._metrics(self.fp_net, self.Ftr, self.Ytr)
                gv, ga = self._metrics(self.fp_net, self.Fte, self.Yte)
                self.hist_fp["tl"].append(fl); self.hist_fp["ta"].append(fa)
                self.hist_fp["vl"].append(gv); self.hist_fp["va"].append(ga)
            if patience > 0:
                if vl < self.best_vl - 1e-6:
                    self.best_vl = vl
                    self.best_epoch = self.epoch
                    self.bad = 0
                    self.best_snap = {"net": self.net.snapshot(),
                                      "fp": self.fp_net.snapshot() if self.fp_net else None}
                else:
                    self.bad += 1
                if self.bad >= patience:
                    self._restore_best()
                    if self.timer.isActive():
                        self.timer.stop()
                        self.play_btn.setText("▶  Train")
                    self.statusBar().showMessage(
                        f"early stop: restored best epoch {self.best_epoch} "
                        f"(test loss {self.best_vl:.3f})", 6000)
                    break
        self.push_all()

    def on_tick(self):
        self.run_epochs(self.speed.value())

    def toggle_play(self, *_):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶  Train")
        else:
            self.timer.start()
            self.play_btn.setText("❚❚  Pause")

    def _metrics(self, net, F, Y):
        if F is None or len(F) == 0:
            return float("nan"), float("nan")
        P = net.predict(F)
        return net.loss(P, Y), float(np.mean((P > 0.5) == (Y > 0.5)))

    def _f1_from(self, P, Y, thr):
        y = Y[:, 0] > 0.5
        yh = P[:, 0] > thr
        tp = float(np.sum(yh & y)); fp = float(np.sum(yh & ~y))
        fn = float(np.sum(~yh & y))
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        if prec and rec and prec + rec > 0:
            return 2 * prec * rec / (prec + rec)
        return None

    def _snap_f1(self):
        if self.Fte is None or not len(self.Fte) or self.net is None:
            return
        P = self.net.predict(self.Fte)
        y = self.Yte[:, 0] > 0.5
        best, bt = -1.0, 0.5
        for t in np.arange(0.02, 0.99, 0.01):
            f1 = self._f1_from(P, self.Yte, float(t)) or 0.0
            if f1 > best:
                best, bt = f1, float(t)
        self.thr_slider.setValue(int(round(bt * 100)))
        self.statusBar().showMessage(
            f"threshold snapped to {bt:.2f} (best test F1 = {best:.3f}) — "
            f"note accuracy barely moved: that's why F1 needs its own knob", 6000)

    # ------------------------------------------------------------ refresh --
    def _trace_acts(self):
        if not self.sel_sample or self.net is None:
            return None
        F = featurize(np.array([[self.sel_sample["x"], self.sel_sample["y"]]]),
                      self.active_features)
        return self.net.trace(F)

    def push_all(self):
        in_labels = [FEATURES[i] for i in self.active_features]
        use_fp = self.fp_net is not None and self.view_combo.currentIndex() == 1
        bnet = self.fp_net if use_fp else self.net

        def probe(xy):
            return float(bnet.predict(featurize(xy, self.active_features))[0, 0])

        def grid(XY):
            return bnet.predict(featurize(XY, self.active_features))

        self.boundary.set_state(bnet, self.Xtr, self.Ytr, self.Xte, self.Yte, probe, grid)
        self.boundary.set_flags(self.chk_test.isChecked(), self.chk_disc.isChecked())
        self.netview.set_state(self.net, in_labels)
        self.netview.set_trace(self._trace_acts())
        self.netview.set_sel(self.sel_neuron)
        if self.sel_sample and self.net is not None:
            F = featurize(np.array([[self.sel_sample["x"], self.sel_sample["y"]]]),
                          self.active_features)
            P = float(self.net.predict(F)[0, 0])
            y = self.sel_sample["label"]
            ok = (P > self.thr) == (y > 0.5)
            tag = "test" if self.sel_sample["test"] else "train"
            self.trace_lbl.setText(
                f"sample #{self.sel_sample['idx']} ({tag}, y = {int(round(y))}) → "
                f"p = {P:.2f} → class {int(P > self.thr)} @thr {self.thr:.2f}  "
                f"{'✓ correct' if ok else '✗ WRONG — inspect its path above'}")
        else:
            self.trace_lbl.setText("no sample selected — 🎯 pick one on the boundary plot")

        if self.net is not None:
            pw = sum(w.size for w in self.net.W)
            pb = sum(b.size for b in self.net.b)
            fpb = (pw + pb) * 4
            tpb = math.ceil(pw * 2 / 8) + pb
            self.d_mem.setText(f"params {pw + pb}  ·  FP32 {fpb / 1024:.2f} KB  →  "
                               f"ternary {tpb / 1024:.2f} KB  ({fpb / max(tpb, 1):.1f}× smaller)")
            adds, macs, pct = self.net.ops()
            self.d_ops.setText(f"inference: {adds} add/sub per sample  ·  "
                               f"{pct:.0f}% of full-precision MACs ({macs})")
            self.d_ram.setText(f"activation RAM {sum(self.net.sizes) * 4} B  ·  "
                               f"16 ternary weights pack into one 32-bit word")

        mode = self.loss_mode.currentText()
        if mode == "Accuracy":
            t, v = self.hist["ta"], self.hist["va"]
            ft = self.hist_fp["ta"] if self.fp_net else None
            fv = self.hist_fp["va"] if self.fp_net else None
        else:
            t, v = self.hist["tl"], self.hist["vl"]
            ft = self.hist_fp["tl"] if self.fp_net else None
            fv = self.hist_fp["vl"] if self.fp_net else None
        self.lossview.set_series(t, v, ft, fv, mode, self.best_epoch)
        self.lbl_best.setText(f"best ep {self.best_epoch}" if self.best_epoch else "best: —")
        self.btn_best.setEnabled(self.best_snap is not None)
        self.update_hist_canvas()

        # confusion + ROC at the current threshold
        if self.Fte is not None and len(self.Fte) and self.net is not None:
            P = self.net.predict(self.Fte)
            y = self.Yte[:, 0] > 0.5
            yh = P[:, 0] > self.thr
            self.confview.set_counts(int(np.sum(yh & y)), int(np.sum(yh & ~y)),
                                     int(np.sum(~yh & y)), int(np.sum(~yh & ~y)),
                                     self.thr)
            self.rocview.set_data(P[:, 0], self.Yte[:, 0], self.thr)
            self.last_f1 = self._f1_from(P, self.Yte, self.thr)
        else:
            self.confview.set_counts(None, None, None, None, self.thr)
            self.rocview.set_data(None, None, self.thr)
            self.last_f1 = None
        self.gradview.set_data(self.hist_g)
        self.neuronview.set_data(self.net, self.sel_neuron, in_labels,
                                 self.active_features, self.epoch)
        self.update_stats()

    def update_hist_canvas(self, *_):
        if self.net is None:
            return
        q = self.net.quantized()
        i = self.hist_combo.currentIndex()
        if i <= 0:
            latent = np.concatenate([w.ravel() for w in self.net.W])
            qb = np.concatenate([w.ravel() for w in q])
            delta = None
            mlayer = -1
        else:
            l = i - 1
            latent = self.net.W[l].ravel()
            qb = q[l].ravel()
            delta = float(self.net.deltas()[l]) if self.net.ternary else None
            mlayer = l
        counts = (int((qb < 0).sum()), int((qb == 0).sum()), int((qb > 0).sum()))
        self.histview.set_data(latent, delta, counts)
        self.matrixview.set_data(self.net, mlayer)

    def update_stats(self):
        h = self.hist
        fmt = lambda s: f"{s[-1]:.3f}" if s else "—"
        self.lbl_epoch_val.setText(str(self.epoch))
        self.lbl_tl_val.setText(fmt(h["tl"]))
        self.lbl_vl_val.setText(fmt(h["vl"]))
        self.lbl_ta_val.setText(f"{h['ta'][-1]*100:.0f} %" if h["ta"] else "—")
        self.lbl_va_val.setText(f"{h['va'][-1]*100:.0f} %" if h["va"] else "—")
        if self.net is not None and h["tl"]:
            msg = (f"epoch {self.epoch}  ·  ternary loss {fmt(h['tl'])} / {fmt(h['vl'])}  ·  "
                   f"acc {h['ta'][-1]*100:.0f}% / {h['va'][-1]*100:.0f}%")
            if getattr(self, "last_f1", None) is not None:
                msg += f"  ·  F1 {self.last_f1:.2f}"
            if self.fp_net is not None and self.hist_fp["tl"]:
                msg += (f"  ·  FP: {fmt(self.hist_fp['tl'])} / {fmt(self.hist_fp['vl'])}")
            self.statusBar().showMessage(msg)

    # ------------------------------------------------------- experiment log --
    def _log_run(self):
        if self.net is None or not self.hist["tl"]:
            QMessageBox.information(self, "Log run",
                                    "Train at least one epoch first.")
            return
        z, tot = self.net.zero_fraction()
        f1 = getattr(self, "last_f1", None)
        entry = dict(
            ts=datetime.now().strftime("%H:%M:%S"),
            dataset=self.data_combo.currentText(),
            arch="-".join(map(str, self.net.sizes)),
            features=",".join(FEATURES[i] for i in self.active_features),
            opt=self._opt_name(), lr=float(self.lr_combo.currentText()),
            l2=float(self.l2_spin.value()), batch=self.batch_combo.currentText(),
            delta=self._delta(), epochs=self.epoch,
            train_loss=round(self.hist["tl"][-1], 4),
            test_loss=round(self.hist["vl"][-1], 4),
            test_acc=round(self.hist["va"][-1], 4),
            test_f1=(round(f1, 4) if f1 is not None else None),
            nonzero_pct=round(100 * (1 - z / tot), 1) if tot else None,
            params=tot + sum(b.size for b in self.net.b),
            seed=self.seed_spin.value())
        self.experiments.append(entry)
        self.statusBar().showMessage(
            f"logged run #{len(self.experiments)} — Tools → Experiment log to "
            f"compare or export CSV", 5000)

    # ------------------------------------------------------- challenge logic --
    def challenge_states(self):
        st = []
        va = self.hist["va"][-1] if self.hist["va"] else None
        vl = self.hist["vl"][-1] if self.hist["vl"] else None
        z, tot = self.net.zero_fraction() if self.net else (0, 1)
        f1 = getattr(self, "last_f1", None)
        fpv = (self.hist_fp["vl"][-1] if (self.fp_net and self.hist_fp["vl"]) else None)
        st.append(va is not None and va >= 0.95)
        st.append(self.data_combo.currentText() == "XOR" and va == 1.0)
        st.append(tot > 0 and z / tot >= 0.5)
        st.append(fpv is not None and vl is not None and vl <= fpv)
        st.append(self.data_combo.currentText() == "Gaussians" and self.epoch == 0
                  and va is not None and va >= 0.9)
        st.append(self.imb.value() <= 30 and f1 is not None and f1 >= 0.9)
        st.append(self.best_epoch is not None)
        st.append(bool(self.flags.get("lr_applied")))
        st.append(bool(self.flags.get("ptq_done")))
        st.append(bool(self.flags.get("cv_done")))
        st.append(self.net is not None and any(self.net.frozen)
                  and va is not None and va >= 0.8)
        st.append(bool(self.flags.get("exported_c")))
        return st

    # ------------------------------------------------------- file menu ----
    def _snapshot_meta(self):
        return dict(sizes=self.net.sizes, activation=self.act_combo.currentText(),
                    optimizer=self._opt_name(), delta_factor=self._delta(),
                    quantize_bias=self.bias_chk.isChecked(),
                    l2=float(self.l2_spin.value()),
                    decay=float(self.decay_spin.value()),
                    hidden=list(self.layer_editor.layers),
                    features=list(self.active_features),
                    dataset=self.data_combo.currentText(),
                    noise=self.noise.value(), n=self.n_spin.value(),
                    ratio=self.ratio.value(), imb=self.imb.value(),
                    seed=self.seed_spin.value(), thr=self.thr,
                    aug=bool(self.aug_chk.isChecked()),
                    aug_sigma=float(self.aug_spin.value()),
                    epoch=self.epoch, t=self.net.t,
                    hist={k: list(v) for k, v in self.hist.items()},
                    hist_fp={k: list(v) for k, v in self.hist_fp.items()},
                    hist_g=[list(g) for g in self.hist_g],
                    best_epoch=self.best_epoch, best_vl=self.best_vl,
                    has_fp=self.fp_net is not None,
                    experiments=list(self.experiments), flags=dict(self.flags))

    def _save_model(self):
        if self.net is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save model", "tnn_model.npz",
                                              "TNN model (*.npz)")
        if not path:
            return
        meta = self._snapshot_meta()
        kw = {}
        for l, (w, b) in enumerate(zip(self.net.W, self.net.b)):
            kw[f"W{l}"] = w; kw[f"b{l}"] = b
        if self.fp_net is not None:
            for l, (w, b) in enumerate(zip(self.fp_net.W, self.fp_net.b)):
                kw[f"fp_W{l}"] = w; kw[f"fp_b{l}"] = b
        if self.best_snap is not None:
            for l, (w, b) in enumerate(zip(self.best_snap["net"][0],
                                           self.best_snap["net"][1])):
                kw[f"bst_W{l}"] = w; kw[f"bst_b{l}"] = b
        np.savez(path, meta=json.dumps(meta), **kw)
        self.statusBar().showMessage(f"model saved → {path}", 4000)

    def _load_model(self):
        if self._loading:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load model", "",
                                              "TNN model (*.npz)")
        if not path:
            return
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(z["meta"].item())
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"Could not read file:\n{e}")
            return
        self._loading = True
        self.data_combo.setCurrentText(meta.get("dataset", "XOR"))
        self.noise.setValue(int(meta.get("noise", 10)))
        self.n_spin.setValue(int(meta.get("n", 300)))
        self.ratio.setValue(int(meta.get("ratio", 50)))
        self.imb.setValue(int(meta.get("imb", 50)))
        self.seed_spin.setValue(int(meta.get("seed", 0)))
        feats = list(meta.get("features", [0, 1]))
        for i, cb in enumerate(self.feat_checks):
            cb.setChecked(i in feats)
        self.active_features = feats
        self.layer_editor.set_layers(meta.get("hidden", [4, 4]))
        self.act_combo.setCurrentText(meta.get("activation", "tanh"))
        self.opt_combo.setCurrentText("Adam" if meta.get("optimizer") == "adam"
                                      else "SGD+momentum")
        self.delta_slider.setValue(int(round(meta.get("delta_factor", 0.7) * 100)))
        self.bias_chk.setChecked(bool(meta.get("quantize_bias", False)))
        self.l2_spin.setValue(float(meta.get("l2", 0.0)))
        self.decay_spin.setValue(float(meta.get("decay", 0.0)))
        self.aug_chk.setChecked(bool(meta.get("aug", False)))
        self.aug_spin.setValue(float(meta.get("aug_sigma", 0.05)))
        self.fp_chk.setChecked(bool(meta.get("has_fp", False)))
        self._loading = False

        self.rng = np.random.default_rng(self.seed_spin.value())
        self.rebuild_net()
        self.regenerate()
        try:
            for l in range(self.net.L):
                self.net.W[l][:] = z[f"W{l}"]
                self.net.b[l][:] = z[f"b{l}"]
            self.net.t = int(meta.get("t", 0))
            if meta.get("has_fp") and self.fp_net is not None:
                for l in range(self.fp_net.L):
                    self.fp_net.W[l][:] = z[f"fp_W{l}"]
                    self.fp_net.b[l][:] = z[f"fp_b{l}"]
            if "bst_W0" in z:
                bw = [z[f"bst_W{l}"] for l in range(self.net.L)]
                bb = [z[f"bst_b{l}"] for l in range(self.net.L)]
                self.best_snap = {"net": (bw, bb, 0), "fp": None}
        except KeyError as e:
            QMessageBox.warning(self, "Load incomplete", f"Missing array: {e}")
        h = meta.get("hist", {})
        self.hist = {k: list(h.get(k, [])) for k in ("tl", "vl", "ta", "va")}
        hf = meta.get("hist_fp", {})
        self.hist_fp = {k: list(hf.get(k, [])) for k in ("tl", "vl", "ta", "va")}
        self.hist_g = [list(g) for g in meta.get("hist_g", [])]
        self.epoch = int(meta.get("epoch", 0))
        self.best_epoch = meta.get("best_epoch")
        self.best_vl = float(meta.get("best_vl", float("inf")))
        self.experiments = list(meta.get("experiments", []))
        self.flags.update(meta.get("flags", {}))
        self.thr_slider.setValue(int(round(float(meta.get("thr", 0.5)) * 100)))
        self.push_all()
        self.statusBar().showMessage(f"model loaded ← {path}", 4000)

    def _export_ternary(self):
        if self.net is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export ternary weights",
                                              "tnn_ternary.npz", "Ternary weights (*.npz)")
        if not path:
            return
        q, bq = self.net.quantized(), self.net.biases_q()
        kw = {f"Wq{l}": q[l].astype(np.int8) for l in range(self.net.L)}
        kw.update({f"b{l}": bq[l].astype(np.float32) for l in range(self.net.L)})
        meta = dict(sizes=self.net.sizes, activation=self.act_combo.currentText(),
                    delta_factor=self._delta(), features=list(self.active_features),
                    threshold=self.thr,
                    format="ternary int8: −1/0/+1")
        np.savez(path, meta=json.dumps(meta), **kw)
        self.statusBar().showMessage(f"ternary weights exported → {path}", 4000)

    def _export_c(self):
        if self.net is None:
            return
        net = self.net
        q, bq = net.quantized(), net.biases_q()
        din = len(self.active_features)
        fexpr = {0: "x[0]", 1: "x[1]", 2: "x[0]*x[0]", 3: "x[1]*x[1]",
                 4: "x[0]*x[1]", 5: "sinf(x[0])", 6: "sinf(x[1])"}
        L = net.L
        s = ["/* Ternary NN — generated by Ternary NN Playground 4.0",
             f"   sizes {net.sizes} · activation {net.activation} · "
             f"decision threshold {self.thr:.2f}",
             "   weights are int8 in {-1,0,+1}: each MAC is an add, sub, or skip. */",
             "#include <math.h>", ""]
        for l in range(L):
            fi, fo = net.sizes[l], net.sizes[l + 1]
            rows = ",\n  ".join("{" + ",".join(str(int(v)) for v in row) + "}"
                                for row in q[l])
            s.append(f"static const signed char W{l}[{fi}][{fo}] = {{\n  {rows}}};")
            s.append("static const float B%d[%d] = {%s};"
                     % (l, fo, ",".join(f"{v:.6f}f" for v in bq[l])))
        s.append("")
        s.append("/* x = raw 2-D input; feature engineering is inlined below. */")
        s.append("float tnn_predict(const float x[2]) {")
        s.append(f"  float a0[{din}];")
        for k, fi_ in enumerate(self.active_features):
            s.append(f"  a0[{k}] = {fexpr[fi_]};")
        prev = "a0"
        for l in range(L):
            fo = net.sizes[l + 1]
            s.append(f"  float z{l + 1}[{fo}]; float a{l + 1}[{fo}];")
            s.append(f"  for (int j = 0; j < {fo}; j++) {{")
            s.append(f"    float s = B{l}[j];")
            s.append(f"    for (int i = 0; i < {net.sizes[l]}; i++) "
                     f"s += (float)W{l}[i][j] * {prev}[i];")
            if l == L - 1:
                s.append(f"    a{l + 1}[j] = 1.0f / (1.0f + expf(-s));")
            elif net.activation == "ReLU":
                s.append(f"    a{l + 1}[j] = s > 0.0f ? s : 0.0f;")
            else:
                s.append(f"    a{l + 1}[j] = tanhf(s);")
            s.append("  }")
            prev = f"a{l + 1}"
        s.append(f"  return {prev}[0];")
        s.append("}")
        s.append("")
        s.append(f"/* classify with YOUR tuned threshold: pred = tnn_predict(x) > "
                 f"{self.thr:.2f}f; */")
        text = "\n".join(s)
        path, _ = QFileDialog.getSaveFileName(self, "Export C inference",
                                              "tnn_inference.c", "C source (*.c)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.flags["exported_c"] = True
        self.statusBar().showMessage(f"C inference code exported → {path}", 4000)

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export boundary PNG",
                                              f"tnn_boundary_epoch{self.epoch}.png",
                                              "PNG image (*.png)")
        if not path:
            return
        self.boundary.grab().save(path, "PNG")
        self.statusBar().showMessage(f"boundary exported → {path}", 4000)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(build_qss(False))
    app.setApplicationName("Ternary NN Playground")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()