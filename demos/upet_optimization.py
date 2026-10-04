# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "upet",
#     "ase>=3.23",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Rolling downhill: structure optimization with UPET + ASE")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ⛰️ Rolling downhill
    ### Structure optimization with ASE and the UPET machine-learning potential

    Every arrangement of atoms has an **energy**. Nature prefers low energy, so molecules and crystals
    settle into shapes where nudging any atom in any direction would only raise the energy.
    Finding that shape on a computer is called **structure optimization** (or *geometry relaxation*),
    and it is the first step of almost every computational chemistry and materials science workflow.

    The idea fits in one sentence: **follow the forces until they vanish.** The force on each atom
    points downhill on the energy landscape, so we move the atoms along the forces, recompute,
    and repeat until every force is tiny.

    In this notebook you will:

    1. **Be the optimizer** by stepping a two-atom molecule downhill by hand.
    2. **Relax a real molecule** that we deliberately scramble, and watch it find its way home.
    3. **Race the optimizers** that ship with ASE against each other.
    4. **Relax a crystal**, including the size of its unit cell, and read off its lattice constant and stiffness.

    Everything is live: change a control and the cells that depend on it re-run automatically.

    /// admonition | ▶️ Only seeing text, no sliders or plots?
    The notebook hasn't been run yet. Switch to the editor view and click **Run all** (or set
    *Settings → Runtime → On startup* to **autorun**), then switch back to app view.
    The first run installs PyTorch and downloads the UPET model, so give it a minute or two.
    ///

    /// details | 🧰 The tools we're using
    - **[ASE](https://wiki.fysik.dtu.dk/ase/)** (the Atomic Simulation Environment) is a Python library for
      building, manipulating, and simulating atomic structures. It provides the `Atoms` object,
      the optimizers, and a common *calculator* interface for computing energies and forces.
    - **[UPET](https://github.com/lab-cosmo/upet)** provides universal machine-learning interatomic
      potentials based on the Point Edge Transformer (PET) architecture. A potential like PET-MAD
      was trained to reproduce quantum-mechanical (DFT) energies and forces across most of the periodic table,
      but runs thousands of times faster, which is what makes this notebook interactive.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    import time
    import warnings

    import numpy as np

    # The UPET calculator also computes a stress for isolated molecules, where it is
    # undefined (NaN) and triggers a harmless RuntimeWarning. Silence just that one.
    warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning,
                            module="metatomic_ase")

    # Downloading the public model weights works fine without a Hugging Face token;
    # hide the "unauthenticated requests to the HF Hub" nag (and only that message).
    import logging

    class _DropHFTokenNag(logging.Filter):
        def filter(self, record):
            return "unauthenticated requests to the HF Hub" not in record.getMessage()

    for _name in ("huggingface_hub", "huggingface_hub.utils._http", "huggingface_hub.file_download"):
        logging.getLogger(_name).addFilter(_DropHFTokenNag())
    warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from ase import Atoms
    from ase.build import bulk, minimize_rotation_and_translation, molecule
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors
    from ase.eos import EquationOfState
    from ase.filters import FrechetCellFilter
    from ase.neighborlist import NeighborList, natural_cutoffs
    from ase.optimize import BFGS, FIRE, LBFGS, GPMin, MDMin
    from ase.units import GPa

    return (
        Atoms,
        BFGS,
        EquationOfState,
        FIRE,
        FrechetCellFilter,
        GPMin,
        GPa,
        LBFGS,
        MDMin,
        NeighborList,
        bulk,
        covalent_radii,
        go,
        jmol_colors,
        make_subplots,
        minimize_rotation_and_translation,
        molecule,
        natural_cutoffs,
        np,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Load the potential

    First we create an ASE **calculator**: the object that, given atom positions, returns the energy
    and forces. With UPET this is a single line:

    ```python
    import torch
    from upet.ase import UPETCalculator

    device = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU if there is one
    calc = UPETCalculator(model="pet-mad-xs", version="1.6.0", device=device)
    ```

    The first time you pick a model its weights are downloaded and cached. Smaller models are faster;
    larger ones are more accurate. **XS** is plenty for exploring.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_picker = mo.ui.dropdown(
        options={
            "PET-MAD XS · fastest": "pet-mad-xs",
            "PET-MAD S · balanced": "pet-mad-s",
            "PET-MAD M · most accurate": "pet-mad-m",
        },
        value="PET-MAD XS · fastest",
        label="**Model**",
    )
    model_picker
    return (model_picker,)


@app.cell(hide_code=True)
def _():
    # Lives in its own cell so loaded models survive re-runs of the loader cell.
    calc_cache = {}
    return (calc_cache,)


@app.cell(hide_code=True)
def _(calc_cache, mo, model_picker, time):
    import os
    import sys

    import torch

    # Importing UPET loads NVIDIA Warp, whose compiled code prints a "Could not find or load the
    # NVIDIA CUDA driver" note straight to the OS-level stderr on machines without a GPU. That's
    # expected here (we fall back to the CPU), so mute stderr just for this import.
    sys.stderr.flush()
    _saved_fd = os.dup(2)
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(_null_fd, 2)
        from upet.ase import UPETCalculator
    finally:
        os.dup2(_saved_fd, 2)
        os.close(_null_fd)
        os.close(_saved_fd)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _key = (model_picker.value, device)
    _t0 = time.perf_counter()
    if _key not in calc_cache:
        with mo.status.spinner(title=f"Loading {model_picker.value} (first use downloads the weights)…"):
            calc_cache[_key] = UPETCalculator(model=model_picker.value, version="1.6.0", device=device)
    calc = calc_cache[_key]

    mo.callout(
        mo.md(
            f"✅ **{model_picker.value}** v1.6.0 is ready on **{device.upper()}** "
            f"({time.perf_counter() - _t0:.1f} s)."
        ),
        kind="success",
    )
    return (calc,)


@app.cell(hide_code=True)
def _(FrechetCellFilter, GPa, calc, np, time):
    def energy_and_forces(atoms):
        """Single-point evaluation: energy (eV) and forces (eV/Å) for a copy of `atoms`."""
        a = atoms.copy()
        a.calc = calc
        return a.get_potential_energy(), a.get_forces()

    def max_force(forces):
        """The largest force on any atom: the number optimizers compare against `fmax`."""
        return float(np.sqrt((np.asarray(forces) ** 2).sum(axis=1)).max())

    def run_optimization(atoms, optimizer_cls, fmax, steps, use_cell_filter=False, **opt_kwargs):
        """Relax `atoms` in place and record a snapshot at every step."""
        atoms.calc = calc
        target = FrechetCellFilter(atoms) if use_cell_filter else atoms
        frames = []

        def record():
            frames.append(
                {
                    "atoms": atoms.copy(),
                    "energy": atoms.get_potential_energy(),
                    "forces": atoms.get_forces().copy(),
                    # For a cell filter this includes the stress "forces" on the cell.
                    "fmax": max_force(target.get_forces()),
                    "pressure": (
                        -atoms.get_stress(voigt=True)[:3].mean() / GPa if atoms.pbc.all() else None
                    ),
                }
            )

        opt = optimizer_cls(target, logfile=None, **opt_kwargs)
        opt.attach(record, interval=1)  # also called once before the first step
        t0 = time.perf_counter()
        converged = opt.run(fmax=fmax, steps=steps)
        return {"frames": frames, "converged": bool(converged), "seconds": time.perf_counter() - t0}

    return energy_and_forces, max_force, run_optimization


@app.cell(hide_code=True)
def _(NeighborList, covalent_radii, go, jmol_colors, natural_cutoffs, np):
    def bonds_of(atoms):
        nl = NeighborList(natural_cutoffs(atoms, mult=1.15), self_interaction=False, bothways=False)
        nl.update(atoms)
        pairs = []
        for i in range(len(atoms)):
            js, offsets = nl.get_neighbors(i)
            for j, off in zip(js, offsets):
                pairs.append((i, j, off))
        return pairs

    # --- Geometry helpers: real 3D meshes, so atoms keep their size (in Å) when you zoom ---
    _LIGHT = dict(ambient=0.45, diffuse=0.8, specular=0.35, roughness=0.45, fresnel=0.1)
    _LIGHTPOS = dict(x=200, y=300, z=1000)

    def _sphere_template(n_lat=14, n_lon=22):
        th = np.linspace(0, np.pi, n_lat)
        ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
        T, P = np.meshgrid(th, ph, indexing="ij")
        verts = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
        faces = []
        for a in range(n_lat - 1):
            for b in range(n_lon):
                i0, i1 = a * n_lon + b, a * n_lon + (b + 1) % n_lon
                i2, i3 = (a + 1) * n_lon + b, (a + 1) * n_lon + (b + 1) % n_lon
                faces += [(i0, i2, i1), (i1, i2, i3)]
        return verts, np.array(faces)

    _SPHERE = _sphere_template()

    def _frame(d):
        d = d / np.linalg.norm(d)
        helper = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(d, helper)
        u /= np.linalg.norm(u)
        return u, np.cross(d, u)

    def _cylinder(p, q, r, n=16):
        u, w = _frame(q - p)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = r * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w))
        verts = np.vstack([p + ring, q + ring])
        faces = []
        for k in range(n):
            a, b = k, (k + 1) % n
            faces += [(a, b, n + a), (b, n + b, n + a)]
        return verts, np.array(faces)

    def _cone(base, tip, r, n=16):
        u, w = _frame(tip - base)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = base + r * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w))
        verts = np.vstack([ring, tip, base])
        faces = []
        for k in range(n):
            a, b = k, (k + 1) % n
            faces += [(a, b, n), (b, a, n + 1)]  # side + base cap
        return verts, np.array(faces)

    class _Mesh:
        """Accumulates many small meshes into one plotly Mesh3d trace (fast to render)."""

        def __init__(self):
            self.v, self.f, self.c, self.t, self.n = [], [], [], [], 0

        def add(self, verts, faces, color, text=""):
            self.v.append(verts)
            self.f.append(faces + self.n)
            self.c += [color] * len(verts)
            self.t += [text] * len(verts)
            self.n += len(verts)

        def trace(self, hover=False):
            if not self.v:
                return None
            v, f = np.vstack(self.v), np.vstack(self.f)
            return go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                vertexcolor=self.c, flatshading=False, lighting=_LIGHT, lightposition=_LIGHTPOS,
                text=self.t if hover else None, hoverinfo="text" if hover else "skip",
                showscale=False, showlegend=False,
            )

    def structure_figure(atoms, forces=None, force_scale=0.6, ranges=None, title=None, height=420):
        """Interactive ball-and-stick 3D view of `atoms` (with optional force arrows) using plotly."""
        pos = atoms.positions
        cell = atoms.cell.array
        z = atoms.numbers
        colors = [f"rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})" for r, g, b in jmol_colors[z]]
        radii = 0.25 + 0.3 * covalent_radii[z]  # Å, so atoms scale with the scene
        fn = np.linalg.norm(forces, axis=1) if forces is not None else None
        fig = go.Figure()

        # Atoms
        balls = _Mesh()
        sv, sf = _SPHERE
        for idx, (sym, p) in enumerate(zip(atoms.get_chemical_symbols(), pos)):
            label = f"{sym} #{idx}<br>x={p[0]:.2f}  y={p[1]:.2f}  z={p[2]:.2f} Å"
            if fn is not None:
                label += f"<br>|F| = {fn[idx]:.3f} eV/Å"
            balls.add(p + radii[idx] * sv, sf, colors[idx], label)

        # Bonds: two half-cylinders, each in its atom's colour
        sticks = _Mesh()
        for a, b, off in bonds_of(atoms):
            p, q = pos[a], pos[b] + off @ cell
            mid = (p + q) / 2
            sticks.add(*_cylinder(p, mid, 0.11), colors[a])
            if not np.any(off):
                sticks.add(*_cylinder(mid, q, 0.11), colors[b])
            else:  # bond crosses the cell boundary: draw a half-stub on each side
                q2 = pos[b]
                p2 = pos[a] - off @ cell
                sticks.add(*_cylinder(q2, (p2 + q2) / 2, 0.11), colors[b])

        # Force arrows: shaft + cone head, skipped for negligible forces
        arrows = _Mesh()
        if forces is not None:
            for p, f, r, mag in zip(pos, forces, radii, fn):
                if mag < 0.01:
                    continue
                d = f / mag
                length = force_scale * mag
                start = p + d * r * 0.8
                tip = start + d * length
                head = min(0.35, 0.45 * length)
                arrows.add(*_cylinder(start, tip - d * head, 0.06), "#e4572e")
                arrows.add(*_cone(tip - d * head, tip, 0.16), "#e4572e")

        for tr in (sticks.trace(), balls.trace(hover=True), arrows.trace()):
            if tr is not None:
                fig.add_trace(tr)

        # Unit cell edges for crystals
        if atoms.pbc.any():
            corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ cell
            cx, cy, cz = [], [], []
            for a in range(8):
                for b in range(a + 1, 8):
                    if bin(a ^ b).count("1") == 1:
                        cx += [corners[a, 0], corners[b, 0], None]
                        cy += [corners[a, 1], corners[b, 1], None]
                        cz += [corners[a, 2], corners[b, 2], None]
            fig.add_trace(
                go.Scatter3d(
                    x=cx, y=cy, z=cz, mode="lines",
                    line=dict(color="rgba(90,120,200,0.7)", width=3, dash="dash"),
                    hoverinfo="skip", showlegend=False,
                )
            )

        axis = dict(visible=False)
        scene = dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data")
        if ranges is not None:
            scene = dict(
                xaxis=dict(visible=False, range=ranges[0]),
                yaxis=dict(visible=False, range=ranges[1]),
                zaxis=dict(visible=False, range=ranges[2]),
                aspectmode="manual",
                aspectratio=dict(zip("xyz", (np.ptp(ranges, axis=1) / np.ptp(ranges, axis=1).max()))),
            )
        scene["camera"] = dict(eye=dict(x=1.0, y=1.0, z=0.75))  # a bit closer than plotly's default
        fig.update_layout(
            scene=scene, height=height, margin=dict(l=0, r=0, t=30 if title else 0, b=0),
            title=dict(text=title, x=0.5, font=dict(size=14)) if title else None,
            uirevision="keep-camera",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def trajectory_ranges(frames, pad=1.3):
        allpos = np.concatenate([f["atoms"].positions for f in frames])
        lo, hi = allpos.min(0) - pad, allpos.max(0) + pad
        return np.stack([lo, hi], axis=1)

    return structure_figure, trajectory_ranges


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1 · Be the optimizer 🧑‍🔬

    The simplest possible "structure" is two atoms joined by a bond. Its energy depends on just one
    number, the bond length $r$, so the energy landscape is a curve you can see.

    The **force** is the negative slope of that curve, $F = -\,\mathrm{d}E/\mathrm{d}r$. A positive force
    pushes the atoms apart (bond too short); a negative force pulls them together (bond too long).

    **Steepest descent**, the most basic optimizer, moves along the force with a fixed step size $\alpha$:

    $$
    r_{\text{new}} = r_{\text{old}} + \alpha\, F(r_{\text{old}})
    $$

    Your turn. Pick a molecule and a starting bond length, then press **Take a step** and watch the
    ball roll. Then try a bigger $\alpha$ and see what goes wrong.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    diatomic_picker = mo.ui.dropdown(
        options={
            "H₂ · hydrogen": "H2",
            "N₂ · nitrogen": "N2",
            "CO · carbon monoxide": "CO",
            "HF · hydrogen fluoride": "HF",
            "Cl₂ · chlorine": "Cl2",
        },
        value="H₂ · hydrogen",
        label="**Molecule**",
    )
    diatomic_picker
    return (diatomic_picker,)


@app.cell(hide_code=True)
def _(Atoms, diatomic_picker, energy_and_forces, molecule, np):
    _ref = molecule(diatomic_picker.value)
    r_ref = _ref.get_distance(0, 1)
    dimer_symbols = _ref.get_chemical_symbols()

    def dimer(r):
        return Atoms(dimer_symbols, positions=[[0, 0, 0], [0, 0, r]])

    def dimer_energy_force(r):
        e, f = energy_and_forces(dimer(r))
        # Force on atom 1 along the bond axis: positive = pushes atoms apart
        return e, float(f[1, 2])

    r_grid = np.linspace(0.65 * r_ref, 2.2 * r_ref, 90)
    _eg = np.array([dimer_energy_force(r)[0] for r in r_grid])
    e_zero = _eg.min()
    e_grid = _eg - e_zero
    r_min = float(r_grid[np.argmin(e_grid)])

    # Refine the minimum with a few Newton steps. The curvature there (eV/Å²) sets the
    # largest stable steepest-descent step: α < 2/k
    _h = 0.005
    for _ in range(4):
        _fp, _fm = dimer_energy_force(r_min + _h)[1], dimer_energy_force(r_min - _h)[1]
        k_curv = -(_fp - _fm) / (2 * _h)
        r_min = r_min + dimer_energy_force(r_min)[1] / k_curv
    alpha_crit = 2.0 / k_curv
    return (
        alpha_crit,
        dimer_energy_force,
        e_grid,
        e_zero,
        k_curv,
        r_grid,
        r_min,
        r_ref,
    )


@app.cell(hide_code=True)
def _(alpha_crit, mo, r_grid, r_ref):
    start_r = mo.ui.slider(
        start=round(float(r_grid[0]), 2), stop=round(float(r_grid[-1]), 2), step=0.01,
        value=round(1.6 * r_ref, 2), label="Starting bond length $r_0$ (Å)", show_value=True, debounce=True,
    )
    alpha = mo.ui.slider(
        steps=[round(alpha_crit * f, 4) for f in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.05, 1.2)],
        value=round(alpha_crit * 0.5, 4), label="Step size $\\alpha$ (Å²/eV)", show_value=True,
    )
    return alpha, start_r


@app.cell(hide_code=True)
def _(mo):
    get_path, set_path = mo.state([])
    return get_path, set_path


@app.cell(hide_code=True)
def _(set_path, start_r):
    # Changing the molecule or the starting point resets the walk.
    set_path([start_r.value])
    return


@app.cell(hide_code=True)
def _(alpha, dimer_energy_force, get_path, mo, np, r_grid, set_path, start_r):
    def _take(n):
        def _cb(v):
            path = list(get_path()) or [start_r.value]
            for _ in range(n):
                r = path[-1]
                _, f = dimer_energy_force(r)
                r_new = float(np.clip(r + alpha.value * f, 0.5 * r_grid[0], 1.3 * r_grid[-1]))
                path.append(r_new)
            set_path(path)
            return (v or 0) + 1
        return _cb

    step_one = mo.ui.button(label="👣 Take a step", on_click=_take(1), kind="success")
    step_ten = mo.ui.button(label="⏩ Take 10 steps", on_click=_take(10))
    reset_walk = mo.ui.button(label="↺ Reset", on_click=lambda v: (set_path([start_r.value]), (v or 0) + 1)[1])
    return reset_walk, step_one, step_ten


@app.cell(hide_code=True)
def _(
    alpha,
    alpha_crit,
    dimer_energy_force,
    e_grid,
    e_zero,
    get_path,
    go,
    k_curv,
    mo,
    np,
    r_grid,
    r_min,
    reset_walk,
    start_r,
    step_one,
    step_ten,
):
    _path = list(get_path()) or [start_r.value]
    _ef = [dimer_energy_force(r) for r in _path]
    _pe = np.array([e for e, _ in _ef]) - e_zero
    _pf = np.array([f for _, f in _ef])
    _r, _e, _f = _path[-1], _pe[-1], _pf[-1]

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=r_grid, y=e_grid, mode="lines", name="UPET energy",
                              line=dict(color="#4c78a8", width=3)))
    _fig.add_trace(go.Scatter(
        x=_path, y=_pe, mode="lines+markers", name="your path",
        line=dict(color="#f58518", width=1.5, dash="dot"),
        marker=dict(size=7, color=list(range(len(_path))), colorscale="Oranges", showscale=False),
        text=[f"step {i}" for i in range(len(_path))],
        hovertemplate="%{text}<br>r = %{x:.3f} Å<br>E = %{y:.3f} eV<extra></extra>",
    ))
    _fig.add_trace(go.Scatter(x=[_r], y=[_e], mode="markers", name="you are here",
                              marker=dict(size=18, color="#f58518", line=dict(color="black", width=2))))
    # Force arrow (scaled for visibility)
    _span = r_grid[-1] - r_grid[0]
    _arrow = np.clip(_f * 0.05, -0.25 * _span, 0.25 * _span)
    if abs(_f) > 1e-3:
        _fig.add_annotation(x=_r + _arrow, y=_e, ax=_r, ay=_e, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor="#e4572e", text="")
    _fig.add_vline(x=r_min, line=dict(color="gray", dash="dash"),
                   annotation_text="minimum", annotation_position="top right")
    _ymax = max(float(e_grid[np.searchsorted(r_grid, r_grid[0] + 0.12 * _span)]), float(_pe.max()) * 1.1, 1.0)
    _fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="bond length r (Å)", yaxis_title="energy above minimum (eV)",
        yaxis_range=[-0.05 * _ymax, _ymax], xaxis_range=[r_grid[0], r_grid[-1]],
        legend=dict(orientation="h", y=-0.18, x=0), paper_bgcolor="rgba(0,0,0,0)",
    )

    _n = len(_path) - 1
    if abs(_f) < 0.05:
        _status = mo.callout(mo.md(f"🎉 **Converged in {_n} steps!** The force is below 0.05 eV/Å, "
                                   f"so you're at the bottom: r = {_r:.3f} Å."), kind="success")
    elif _n >= 2 and np.sign(_pf[-1]) != np.sign(_pf[-2]):
        _status = mo.callout(mo.md("↔️ **Overshooting.** The force flipped sign, so you jumped past the minimum. "
                                   "Small overshoots still converge; big ones don't."), kind="warn")
    else:
        _status = mo.callout(mo.md(f"The force is **{_f:+.2f} eV/Å**: "
                                   f"{'pushing the atoms apart' if _f > 0 else 'pulling the atoms together'}. "
                                   "Keep stepping!"), kind="info")

    _controls = mo.vstack([
        start_r, alpha,
        mo.hstack([step_one, step_ten, reset_walk], justify="start"),
        mo.hstack([
            mo.stat(f"{_n}", label="steps taken", bordered=True),
            mo.stat(f"{_r:.3f} Å", label="bond length", bordered=True),
            mo.stat(f"{_f:+.2f}", label="force (eV/Å)", bordered=True),
        ], widths="equal"),
        _status,
        mo.accordion({
            "💡 Why does a big α blow up?": mo.md(
                f"Near the minimum the curve looks like a parabola, $E \\approx \\tfrac12 k (r-r_\\mathrm{{min}})^2$, "
                f"with stiffness $k \\approx {k_curv:.0f}$ eV/Å². One steepest-descent step multiplies the distance "
                f"from the minimum by $(1-\\alpha k)$. If $\\alpha > 2/k \\approx {alpha_crit:.4f}$ Å²/eV, that factor "
                "is bigger than 1 in size and every step lands *further* away. Stiff bonds (N₂, CO) need smaller "
                "steps than soft ones (H₂, Cl₂). Real optimizers avoid this trap by learning the curvature as they go."
            )
        }),
    ])
    mo.hstack([_controls, _fig], widths=[2, 3], align="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2 · Relax a real molecule 🧪

    With $N$ atoms the energy depends on $3N$ coordinates at once, so we can no longer draw the
    landscape. The recipe is unchanged, though, and ASE packages it into a few lines:

    ```python
    from ase.build import molecule
    from ase.optimize import BFGS

    atoms = molecule("H2O")                # 1. build a structure
    atoms.rattle(stdev=0.2)                # 2. (here) scramble it on purpose
    atoms.calc = calc                      # 3. attach the UPET calculator
    BFGS(atoms).run(fmax=0.02)             # 4. step until every force < 0.02 eV/Å
    ```

    `fmax` is the **convergence criterion**: the run stops once the largest force on any atom drops
    below it. Below, we shake a molecule with random displacements and let an optimizer bring it back.
    Drag the **step slider** under the plots to replay the trajectory, and hover over atoms to see their forces.
    """)
    return


@app.cell(hide_code=True)
def _(BFGS, FIRE, GPMin, LBFGS, MDMin, mo):
    optimizers = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE, "MDMin": MDMin, "GPMin": GPMin}
    mol_picker = mo.ui.dropdown(
        options={
            "Water · H₂O": "H2O",
            "Ammonia · NH₃": "NH3",
            "Methane · CH₄": "CH4",
            "Formaldehyde · H₂CO": "H2CO",
            "Carbon dioxide · CO₂": "CO2",
            "Ethylene · C₂H₄": "C2H4",
            "Formic acid · HCOOH": "HCOOH",
            "Ethanol · C₂H₅OH": "CH3CH2OH",
            "Acetic acid · CH₃COOH": "CH3COOH",
            "Benzene · C₆H₆": "C6H6",
        },
        value="Ethanol · C₂H₅OH",
        label="**Molecule**",
    )
    rattle_amp = mo.ui.slider(0.0, 0.4, step=0.02, value=0.2, label="Shake strength (Å)", show_value=True, debounce=True)
    rattle_seed = mo.ui.number(start=0, stop=999, value=7, label="Random seed")
    opt_picker = mo.ui.dropdown(options=list(optimizers), value="BFGS", label="**Optimizer**")
    fmax_slider = mo.ui.slider(
        steps=[0.2, 0.1, 0.05, 0.02, 0.01, 0.005], value=0.02,
        label="Convergence `fmax` (eV/Å)", show_value=True,
    )
    max_steps = mo.ui.slider(10, 500, step=10, value=200, label="Max steps", show_value=True, debounce=True)

    mo.hstack(
        [
            mo.vstack([mol_picker, rattle_amp, rattle_seed]),
            mo.vstack([opt_picker, fmax_slider, max_steps]),
        ],
        widths="equal",
    )
    return (
        fmax_slider,
        max_steps,
        mol_picker,
        opt_picker,
        optimizers,
        rattle_amp,
        rattle_seed,
    )


@app.cell(hide_code=True)
def _(mol_picker, molecule, rattle_amp, rattle_seed):
    mol_reference = molecule(mol_picker.value)  # tabulated (G2) reference geometry
    mol_start = mol_reference.copy()
    mol_start.rattle(stdev=rattle_amp.value, seed=int(rattle_seed.value))
    return mol_reference, mol_start


@app.cell(hide_code=True)
def _(
    fmax_slider,
    max_steps,
    mo,
    mol_start,
    opt_picker,
    optimizers,
    run_optimization,
):
    with mo.status.spinner(title=f"Relaxing with {opt_picker.value}…"):
        mol_result = run_optimization(
            mol_start.copy(), optimizers[opt_picker.value], fmax=fmax_slider.value, steps=max_steps.value
        )
    return (mol_result,)


@app.cell(hide_code=True)
def _(mo, mol_result):
    frame_slider = mo.ui.slider(
        start=0, stop=len(mol_result["frames"]) - 1, step=1, value=len(mol_result["frames"]) - 1,
        label="🎞️ Step", show_value=True, full_width=True,
    )
    show_forces = mo.ui.switch(value=True, label="Show force arrows")
    return frame_slider, show_forces


@app.cell(hide_code=True)
def _(
    fmax_slider,
    frame_slider,
    go,
    make_subplots,
    mo,
    mol_result,
    show_forces,
    structure_figure,
    trajectory_ranges,
):
    _frames = mol_result["frames"]
    _i = frame_slider.value
    _fr = _frames[_i]
    _e0 = _frames[-1]["energy"]

    _view = structure_figure(
        _fr["atoms"], forces=_fr["forces"] if show_forces.value else None,
        ranges=trajectory_ranges(_frames), title=f"step {_i}", height=430,
    )

    _steps = list(range(len(_frames)))
    _conv = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    _conv.add_trace(go.Scatter(x=_steps, y=[f["energy"] - _e0 for f in _frames], mode="lines+markers",
                               marker=dict(size=5), line=dict(color="#4c78a8"), name="energy"), row=1, col=1)
    _conv.add_trace(go.Scatter(x=_steps, y=[f["fmax"] for f in _frames], mode="lines+markers",
                               marker=dict(size=5), line=dict(color="#e4572e"), name="max force"), row=2, col=1)
    _conv.add_hline(y=fmax_slider.value, line=dict(color="green", dash="dash"), row=2, col=1,
                    annotation_text="fmax target", annotation_position="top right")
    _conv.add_vline(x=_i, line=dict(color="black", width=1))
    _conv.update_yaxes(title_text="E − E_final (eV)", type="log", row=1, col=1)
    _conv.update_yaxes(title_text="max |F| (eV/Å)", type="log", row=2, col=1)
    _conv.update_xaxes(title_text="optimizer step", row=2, col=1)
    # log axis can't show the final zero; nudge it
    _conv.data[0].y = [max(y, 1e-5) for y in _conv.data[0].y]
    _conv.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)")

    mo.vstack([
        mo.hstack([_view, _conv], widths="equal"),
        mo.hstack([frame_slider, show_forces], widths=[5, 1], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _(
    fmax_slider,
    minimize_rotation_and_translation,
    mo,
    mol_reference,
    mol_result,
    mol_start,
    np,
    opt_picker,
):
    _frames = mol_result["frames"]
    _final = _frames[-1]["atoms"]

    def _rmsd(a, ref):
        a = a.copy()
        minimize_rotation_and_translation(ref, a)
        return float(np.sqrt(((a.positions - ref.positions) ** 2).sum(1).mean()))

    _rmsd_start = _rmsd(mol_start, mol_reference)
    _rmsd_final = _rmsd(_final, mol_reference)

    if mol_result["converged"]:
        _verdict = mo.callout(mo.md(
            f"✅ **{opt_picker.value} converged** in {len(_frames) - 1} steps: every force is below "
            f"{fmax_slider.value} eV/Å."), kind="success")
    else:
        _verdict = mo.callout(mo.md(
            f"⏳ **Not converged** after {len(_frames) - 1} steps. Raise *Max steps*, loosen `fmax`, "
            "or try a different optimizer."), kind="warn")

    # Bond lengths: shaken vs relaxed vs tabulated reference
    _syms = mol_reference.get_chemical_symbols()
    _rows = []
    from ase.neighborlist import natural_cutoffs as _nc, NeighborList as _NL
    _nl = _NL(_nc(mol_reference, mult=1.15), self_interaction=False, bothways=False)
    _nl.update(mol_reference)
    for _a in range(len(mol_reference)):
        for _b in _nl.get_neighbors(_a)[0]:
            _rows.append({
                "bond": f"{_syms[_a]}{_a}–{_syms[_b]}{_b}",
                "shaken (Å)": round(mol_start.get_distance(_a, _b), 3),
                "relaxed (Å)": round(_final.get_distance(_a, _b), 3),
                "reference (Å)": round(mol_reference.get_distance(_a, _b), 3),
            })

    mo.vstack([
        _verdict,
        mo.hstack([
            mo.stat(f"{len(_frames) - 1}", label="optimizer steps", bordered=True),
            mo.stat(f"{_frames[0]['energy'] - _frames[-1]['energy']:.2f} eV", label="energy released",
                    caption="how far downhill we rolled", bordered=True),
            mo.stat(f"{_frames[-1]['fmax']:.3f}", label="final max force (eV/Å)", bordered=True),
            mo.stat(f"{_rmsd_start:.2f} → {_rmsd_final:.2f} Å", label="distance from reference shape",
                    caption="RMSD after alignment", direction="decrease", target_direction="decrease", bordered=True),
            mo.stat(f"{mol_result['seconds']:.2f} s", label="wall time", bordered=True),
        ], widths="equal"),
        mo.accordion({
            "📏 Bond lengths: shaken vs relaxed vs reference": mo.vstack([
                mo.md("The *reference* column is the tabulated G2 geometry shipped with ASE. The relaxed "
                      "bonds won't match it perfectly: the reference and PET-MAD use different levels of theory, "
                      "so differences of ~0.01 Å are expected."),
                mo.ui.table(_rows, selection=None, pagination=False),
            ]),
            "🤔 Didn't return to the reference shape?": mo.md(
                "A strong shake can knock the molecule over a hill into a different valley: a different "
                "**conformer** (e.g. a rotated OH group in ethanol) or even broken bonds. A local optimizer "
                "only ever goes downhill, so it finds the *nearest* minimum, not necessarily the lowest one. "
                "Finding the global minimum needs different tools (conformer searches, basin hopping, …)."
            ),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3 · The optimizer race 🏁

    Steepest descent is simple but slow. ASE ships smarter algorithms that differ in how they decide
    **where to step next**:

    | Optimizer | Idea in one line |
    |---|---|
    | **BFGS** | Builds up an estimate of the landscape's curvature (the Hessian) and jumps toward the bottom of the local bowl. The usual default. |
    | **LBFGS** | Same idea, but remembers only the last few steps. Uses far less memory for big systems. |
    | **FIRE** | Rolls like a ball with friction (molecular dynamics), stopping whenever it starts going uphill. Very robust. |
    | **MDMin** | Also dynamics-based: moves atoms like a ball and kills the velocity when it points uphill. |
    | **GPMin** | Fits a Gaussian-process surrogate to the energies seen so far and minimizes that. Few steps, but each one costs more. |

    All five start from **the same shaken molecule you set up above**, so change the molecule or
    the shake and the race re-runs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    race_button = mo.ui.run_button(label="🏁 Start the race", kind="success")
    race_button
    return (race_button,)


@app.cell(hide_code=True)
def _(fmax_slider, mo, mol_start, optimizers, race_button, run_optimization):
    mo.stop(not race_button.value, mo.md("*Press the button to race all five optimizers.*"))
    race = {}
    for _name, _cls in mo.status.progress_bar(list(optimizers.items()), title="Racing…", remove_on_exit=True):
        race[_name] = run_optimization(mol_start.copy(), _cls, fmax=fmax_slider.value, steps=300)
    return (race,)


@app.cell(hide_code=True)
def _(fmax_slider, go, make_subplots, mo, race):
    _colors = {"BFGS": "#4c78a8", "LBFGS": "#72b7b2", "FIRE": "#e4572e", "MDMin": "#f58518", "GPMin": "#54a24b"}
    _emin = min(r["frames"][-1]["energy"] for r in race.values())
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("max force (eV/Å)", "energy above best (eV)"))
    for _name, _res in race.items():
        _fr = _res["frames"]
        _x = list(range(len(_fr)))
        _fig.add_trace(go.Scatter(x=_x, y=[f["fmax"] for f in _fr], name=_name, mode="lines",
                                  line=dict(color=_colors[_name], width=2.5), legendgroup=_name), 1, 1)
        _fig.add_trace(go.Scatter(x=_x, y=[max(f["energy"] - _emin, 1e-5) for f in _fr], name=_name,
                                  mode="lines", line=dict(color=_colors[_name], width=2.5),
                                  legendgroup=_name, showlegend=False), 1, 2)
    _fig.add_hline(y=fmax_slider.value, line=dict(color="green", dash="dash"), row=1, col=1)
    _fig.update_yaxes(type="log")
    _fig.update_xaxes(title_text="step")
    _fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", y=-0.2))

    _ranked = sorted(race.items(), key=lambda kv: (not kv[1]["converged"], len(kv[1]["frames"])))
    _medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    _table = [
        {
            "": _medals[_k] if _res["converged"] else "❌",
            "optimizer": _name,
            "steps": len(_res["frames"]) - 1,
            "converged": "yes" if _res["converged"] else "no",
            "final E − best (meV)": round(1000 * (_res["frames"][-1]["energy"] - _emin), 2),
            "time (s)": round(_res["seconds"], 2),
        }
        for _k, (_name, _res) in enumerate(_ranked)
    ]
    mo.vstack([
        _fig,
        mo.ui.table(_table, selection=None, pagination=False),
        mo.callout(mo.md(
            "**Reading the race.** Fewer steps means fewer calls to the potential, which is what matters when "
            "each call is an expensive DFT calculation. Quasi-Newton methods (BFGS, LBFGS) usually win on smooth "
            "landscapes; FIRE is slower but hard to break when the start is very rough. MDMin uses a fixed time step "
            "and can crawl or stall near the end. Final energies can differ slightly when optimizers land in "
            "different nearby minima."
        ), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4 · Relax a crystal 💎

    In a crystal the atoms repeat forever in a **unit cell**. Besides the atom positions, the
    *shape and size of the cell* can also be wrong. The cell feels a "force" too: the **stress**
    (pressure). A squeezed crystal pushes outward; a stretched one pulls inward.

    ASE lets an optimizer move the cell by wrapping the atoms in a **filter** that bundles the
    stress together with the atomic forces:

    ```python
    from ase.build import bulk
    from ase.filters import FrechetCellFilter

    atoms = bulk("Cu", "fcc", a=3.80, cubic=True)   # deliberately strained
    atoms.calc = calc
    BFGS(FrechetCellFilter(atoms)).run(fmax=0.01)   # relax positions AND cell
    print(atoms.cell.cellpar())                     # relaxed lattice constant
    ```

    Pick a crystal and strain it. Then compare relaxing **atoms only** with **atoms + cell**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    crystals = {
        "Copper · fcc metal": dict(name="Cu", crystalstructure="fcc", a=3.615, B=140),
        "Aluminium · fcc metal": dict(name="Al", crystalstructure="fcc", a=4.050, B=76),
        "Tungsten · bcc metal": dict(name="W", crystalstructure="bcc", a=3.165, B=310),
        "Silicon · diamond semiconductor": dict(name="Si", crystalstructure="diamond", a=5.431, B=98),
        "Table salt · NaCl rock salt": dict(name="NaCl", crystalstructure="rocksalt", a=5.640, B=25),
        "Magnesium oxide · MgO rock salt": dict(name="MgO", crystalstructure="rocksalt", a=4.212, B=160),
    }
    crystal_picker = mo.ui.dropdown(options=list(crystals), value="Silicon · diamond semiconductor",
                                    label="**Crystal**")
    strain = mo.ui.slider(-8, 8, step=1, value=5, label="Starting strain (%)", show_value=True, debounce=True)
    relax_mode = mo.ui.radio(options=["Atoms only", "Atoms + cell"], value="Atoms + cell", label="**Relax**",
                             inline=True)
    crystal_rattle = mo.ui.switch(value=False, label="Also shake the atoms (0.05 Å)")
    mo.hstack([mo.vstack([crystal_picker, strain]), mo.vstack([relax_mode, crystal_rattle])], widths="equal")
    return crystal_picker, crystal_rattle, crystals, relax_mode, strain


@app.cell(hide_code=True)
def _(
    BFGS,
    bulk,
    crystal_picker,
    crystal_rattle,
    crystals,
    mo,
    relax_mode,
    run_optimization,
    strain,
):
    crystal_info = crystals[crystal_picker.value]
    a_start = crystal_info["a"] * (1 + strain.value / 100)
    crystal_start = bulk(crystal_info["name"], crystal_info["crystalstructure"], a=a_start, cubic=True)
    if crystal_rattle.value:
        crystal_start.rattle(stdev=0.05, seed=1)

    with mo.status.spinner(title="Relaxing the crystal…"):
        crystal_result = run_optimization(
            crystal_start.copy(), BFGS, fmax=0.005, steps=200,
            use_cell_filter=(relax_mode.value == "Atoms + cell"),
        )
    return a_start, crystal_info, crystal_result


@app.cell(hide_code=True)
def _(EquationOfState, GPa, bulk, crystal_info, crystal_result, energy_and_forces, np):
    # Energy-volume curve of the ideal crystal, used as the backdrop for the optimizer's path
    _n = len(crystal_result["frames"][0]["atoms"])
    _a_vals = crystal_info["a"] * np.linspace(0.90, 1.10, 21)
    eos_v, eos_e = [], []
    for _a in _a_vals:
        _at = bulk(crystal_info["name"], crystal_info["crystalstructure"], a=_a, cubic=True)
        eos_v.append(_at.get_volume() / _n)
        eos_e.append(energy_and_forces(_at)[0] / _n)
    eos_v, eos_e = np.array(eos_v), np.array(eos_e)

    _eos = EquationOfState(eos_v, eos_e, eos="birchmurnaghan")
    eos_v0, eos_e0, _B = _eos.fit()
    bulk_modulus_gpa = _B / GPa
    # cubic conventional cell: a = (V_cell)^(1/3)
    a_eos = (eos_v0 * _n) ** (1 / 3)
    return a_eos, bulk_modulus_gpa, eos_e, eos_v


@app.cell(hide_code=True)
def _(mo, crystal_result):
    crystal_frame = mo.ui.slider(
        start=0, stop=len(crystal_result["frames"]) - 1, step=1, value=len(crystal_result["frames"]) - 1,
        label="🎞️ Step", show_value=True, full_width=True,
    )
    return (crystal_frame,)


@app.cell(hide_code=True)
def _(
    a_eos,
    a_start,
    bulk_modulus_gpa,
    crystal_frame,
    crystal_info,
    crystal_result,
    eos_e,
    eos_v,
    go,
    mo,
    relax_mode,
    structure_figure,
):
    _frames = crystal_result["frames"]
    _i = crystal_frame.value
    _n = len(_frames[0]["atoms"])
    _pv = [f["atoms"].get_volume() / _n for f in _frames]
    _pe = [f["energy"] / _n for f in _frames]

    _eosfig = go.Figure()
    _eosfig.add_trace(go.Scatter(x=eos_v, y=eos_e, mode="lines", name="energy vs volume (ideal crystal)",
                                 line=dict(color="#4c78a8", width=3)))
    _eosfig.add_trace(go.Scatter(x=_pv, y=_pe, mode="lines+markers", name="optimizer path",
                                 line=dict(color="#f58518", dash="dot"), marker=dict(size=7, color="#f58518")))
    _eosfig.add_trace(go.Scatter(x=[_pv[_i]], y=[_pe[_i]], mode="markers", name=f"step {_i}",
                                 marker=dict(size=16, color="#f58518", line=dict(color="black", width=2))))
    _eosfig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="volume per atom (Å³)", yaxis_title="energy per atom (eV)",
                          legend=dict(orientation="h", y=1.1, x=0))

    _view = structure_figure(_frames[_i]["atoms"], forces=_frames[_i]["forces"], height=400,
                             title=f"step {_i} · a = {_frames[_i]['atoms'].cell.lengths()[0]:.3f} Å")

    _a_final = _frames[-1]["atoms"].cell.lengths()[0]
    _p_start, _p_final = _frames[0]["pressure"], _frames[-1]["pressure"]
    _nsteps = len(_frames) - 1

    if relax_mode.value == "Atoms only" and abs(_p_final) > 0.5:
        _note = mo.callout(mo.md(
            f"🔒 **The cell was frozen**, so the lattice constant is still {_a_final:.3f} Å and the crystal is "
            f"left under **{_p_final:+.1f} GPa** of pressure. In a perfect crystal, symmetry makes every atomic "
            "force exactly zero, so the optimizer stops immediately even though the structure is far from "
            "relaxed. Switch to **Atoms + cell** to fix it."), kind="warn")
    else:
        _err = 100 * (_a_final - crystal_info["a"]) / crystal_info["a"]
        _note = mo.callout(mo.md(
            f"📐 Relaxed lattice constant **{_a_final:.3f} Å** vs. experiment ≈ {crystal_info['a']:.3f} Å "
            f"({_err:+.1f}%). Errors of 1–2% are typical: the model learned from DFT, which has its own "
            "systematic errors, and experiments include thermal expansion and zero-point motion."), kind="success")

    mo.vstack([
        mo.hstack([_view, _eosfig], widths="equal"),
        crystal_frame,
        mo.hstack([
            mo.stat(f"{a_start:.3f} → {_a_final:.3f} Å", label="lattice constant a", bordered=True),
            mo.stat(f"{_p_start:+.1f} → {_p_final:+.2f} GPa", label="pressure",
                    caption="positive = squeezed, wants to expand", bordered=True),
            mo.stat(f"{_nsteps}", label="optimizer steps", bordered=True),
            mo.stat(f"{bulk_modulus_gpa:.0f} GPa", label="bulk modulus (from E–V fit)",
                    caption=f"experiment ≈ {crystal_info['B']} GPa", bordered=True),
        ], widths="equal"),
        _note,
        mo.accordion({
            "📈 What is the blue curve, and the bulk modulus?": mo.md(
                f"The blue curve is the energy of the perfect crystal at 21 uniformly scaled volumes, "
                "a classic **equation of state** (E–V) scan. Its bottom is the equilibrium volume, and the "
                "optimizer's path should end there. Fitting a Birch–Murnaghan equation gives the equilibrium "
                f"lattice constant (a ≈ {a_eos:.3f} Å here, agreeing with the relaxation) and the **bulk modulus** "
                "$B = V\\,\\mathrm{d}^2E/\\mathrm{d}V^2$, which measures how hard the crystal is to compress. "
                "Diamond-like and refractory materials are stiff; salts are soft. If you shook the atoms, the "
                "first points of the path sit *above* the curve: that extra energy comes from atoms being off "
                "their ideal sites, and the optimizer removes it alongside the strain."
            ),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧭 Recap

    - An optimizer **follows the forces downhill** until the largest force falls below `fmax`.
    - The step size matters: steepest descent with too large a step **overshoots and diverges**.
      Smarter optimizers (BFGS, FIRE, …) adapt automatically.
    - Local optimizers find the **nearest minimum**, which isn't always the global one.
    - For crystals, the **cell** must be relaxed too, using a filter like `FrechetCellFilter`. Symmetry alone
      can make atomic forces vanish in a badly strained crystal.
    - A universal ML potential like **UPET / PET-MAD** makes all of this fast enough to explore interactively.
      For publication-quality numbers, it's common to refine the final structure with DFT.

    ### 🧪 Things to try
    1. Set the shake strength to **0.4 Å** on ethanol and check the bond table: did a bond break or the molecule change conformer?
    2. In the race, tighten `fmax` to **0.005** and see which optimizer struggles most near the end.
    3. Switch the model to **PET-MAD S** and compare the lattice constants and bulk moduli in section 4.
    4. Compress **NaCl** by −8% and watch how hard it pushes back compared with **tungsten**.

    ### 📋 Cheat sheet
    ```python
    from ase.build import molecule, bulk
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter
    from ase.io import write
    from upet.ase import UPETCalculator
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = UPETCalculator(model="pet-mad-s", version="1.6.0", device=device)

    # Molecule: relax atomic positions
    mol = molecule("CH3CH2OH")
    mol.calc = calc
    BFGS(mol, trajectory="ethanol.traj").run(fmax=0.02)   # view with: ase gui ethanol.traj

    # Crystal: relax positions + cell
    si = bulk("Si", "diamond", a=5.6, cubic=True)
    si.calc = calc
    BFGS(FrechetCellFilter(si)).run(fmax=0.01)
    print(si.cell.cellpar())
    write("si_relaxed.cif", si)
    ```

    *UPET & PET-MAD: [github.com/lab-cosmo/upet](https://github.com/lab-cosmo/upet) ·
    ASE optimizers: [ASE documentation](https://wiki.fysik.dtu.dk/ase/ase/optimize.html)*
    """)
    return


if __name__ == "__main__":
    app.run()
