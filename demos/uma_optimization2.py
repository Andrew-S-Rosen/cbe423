Claude
Claude Desktop (Windows), Connected














Rolling downhill · PY
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "fairchem-core==2.24.0",
#     "huggingface-hub",
#     "ase>=3.26",
#     "numpy",
#     "plotly",
# ]
# ///
 
import marimo
 
__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Rolling downhill: structure optimization with UMA + ASE")
 
 
@app.cell(hide_code=True)
def _():
    import marimo as mo
 
    return (mo,)
 
 
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ⛰️ Rolling downhill
    ### Structure optimization with ASE and the UMA machine-learning potential
 
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
    5. **Land molecules on a catalyst surface**: find where they stick, drop them anywhere and see where they roll,
       and map the whole energy landscape an adsorbate feels.
 
    Everything is live: change a control and the cells that depend on it re-run automatically.
 
    /// admonition | ▶️ Only seeing text, no sliders or plots?
    The notebook hasn't been run yet. Switch to the editor view and click **Run all** (or set
    *Settings → Runtime → On startup* to **autorun**), then switch back to app view.
    The first run installs the packages and downloads the UMA model, so give it a minute or two.
    You also need a (free) Hugging Face token with access to UMA: see section 0.
    ///
 
    /// details | 🧰 The tools we're using
    - **[ASE](https://wiki.fysik.dtu.dk/ase/)** (the Atomic Simulation Environment) is a Python library for
      building, manipulating, and simulating atomic structures. It provides the `Atoms` object,
      the optimizers, and a common *calculator* interface for computing energies and forces.
    - **[UMA](https://huggingface.co/facebook/UMA)** (Universal Model for Atoms, from Meta's
      [FAIR Chemistry](https://github.com/facebookresearch/fairchem) team) is a universal machine-learning
      interatomic potential. It was trained on hundreds of millions of quantum-mechanical (DFT) calculations
      spanning molecules, materials, catalysts, and more, and it reproduces DFT energies and forces across most
      of the periodic table while running thousands of times faster. That speed is what makes this notebook
      interactive. We use the small model, **UMA-S 1.2.1** (`uma-s-1p2p1`).
    ///
    """)
    return
 
 
@app.cell(hide_code=True)
def _(mo):
    # molab (and a plain `marimo edit`) ignores the script header at the top of this file, so install
    # anything that's missing (or the wrong fairchem version) here. With `marimo edit --sandbox`
    # everything is already present.
    import importlib as _importlib
    import importlib.util as _ilu
    import shutil as _shutil
    import subprocess as _subprocess
    import sys as _sys
    from importlib.metadata import PackageNotFoundError as _PNF
    from importlib.metadata import version as _version
 
    FAIRCHEM_VERSION = "2.24.0"
    _requirements = {"ase": "ase>=3.26", "fairchem.core": f"fairchem-core=={FAIRCHEM_VERSION}", "plotly": "plotly"}
 
    def _needs_install(mod, req):
        if _ilu.find_spec(mod.split(".")[0]) is None:
            return True
        if req.startswith("fairchem-core=="):
            try:
                return _version("fairchem-core") != FAIRCHEM_VERSION
            except _PNF:
                return True
        return False
 
    _missing = [_req for _mod, _req in _requirements.items() if _needs_install(_mod, _req)]
    if _missing:
        _uv = _shutil.which("uv")
        _cmd = ([_uv, "pip", "install", "--python", _sys.executable] if _uv
                else [_sys.executable, "-m", "pip", "install"]) + _missing
        with mo.status.spinner(title=f"Installing {', '.join(_missing)} (first run only, ~1–2 min)…"):
            _proc = _subprocess.run(_cmd, capture_output=True, text=True)
        mo.stop(_proc.returncode != 0, mo.callout(mo.vstack([
            mo.md("❌ **Installing the packages failed.** Restart the kernel and try again; the log is below."),
            mo.plain_text(_proc.stderr[-3000:]),
        ]), kind="danger"))
        _importlib.invalidate_caches()
    deps_ready = True
    return (deps_ready,)
 
 
@app.cell(hide_code=True)
def _(deps_ready):
    assert deps_ready  # run after the install cell above
    import time
 
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
 
    from ase import Atoms
    from ase.build import add_adsorbate, bulk, fcc100, fcc111, minimize_rotation_and_translation, molecule
    from ase.constraints import FixAtoms, FixCartesian
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors
    from ase.eos import EquationOfState
    from ase.filters import FrechetCellFilter
    from ase.geometry import find_mic
    from ase.neighborlist import NeighborList, natural_cutoffs
    from ase.optimize import BFGS, FIRE, LBFGS, GPMin, MDMin
    from ase.units import GPa
 
    return (
        Atoms,
        BFGS,
        EquationOfState,
        FIRE,
        FixAtoms,
        FixCartesian,
        FrechetCellFilter,
        GPMin,
        GPa,
        LBFGS,
        MDMin,
        NeighborList,
        add_adsorbate,
        bulk,
        covalent_radii,
        fcc100,
        fcc111,
        find_mic,
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
    and forces. With UMA it takes two lines: load the model once (a *predictor*), then wrap it in a
    calculator for the kind of system you are simulating.
 
    ```python
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
 
    device = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU if there is one
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device=device)
    calc = FAIRChemCalculator(predictor, task_name="omol")    # molecules
    ```
 
    UMA is one network with several **task heads**, one per family of training data, each matching a
    different DFT setup. We use three of them:
 
    | `task_name` | Trained on | Used here for |
    |---|---|---|
    | `"omol"` | molecules (ωB97M-V hybrid DFT) | sections 1–3. Needs the total `charge` and spin multiplicity `spin` in `atoms.info` (default: neutral singlet). |
    | `"omat"` | inorganic crystals (PBE / PBE+U) | section 4 |
    | `"oc20"` | adsorbates on catalyst surfaces (RPBE) | section 5 |
 
    The notebook picks the head for you: periodic systems get `omat`, isolated molecules get `omol`,
    and the surface section asks for `oc20` explicitly.
    """)
    return
 
 
@app.cell(hide_code=True)
def _(mo):
    hf_token = mo.ui.text(
        kind="password",
        placeholder="hf_…",
        label="**Hugging Face token**",
        full_width=True,
    )
    mo.vstack([
        mo.md(
            "The UMA weights are free but **gated**: Hugging Face only lets you download them after you accept "
            "Meta's license. You only need to do this once."
        ),
        mo.accordion({
            "🔑 How do I get a token?": mo.md(
                """
    1. Create a free account at [huggingface.co/join](https://huggingface.co/join) (or log in).
    2. Go to [huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA), fill in the form, and accept
       the FAIR Chemistry License. **Approval is manual and can take from minutes to a day or two**, so do this
       before class. You'll get an email when you're approved.
    3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):
       **Create new token → Read** is enough. Copy it (it starts with `hf_`).
    4. Paste it in the box below. It stays in this session only and is not saved in the notebook.
    """
            )
        }),
        hf_token,
    ])
    return (hf_token,)
 
 
@app.cell(hide_code=True)
def _():
    # Lives in its own cell so the loaded model survives re-runs of the loader cell.
    predictor_cache = {}
    return (predictor_cache,)
 
 
@app.cell(hide_code=True)
def _(hf_token, mo, predictor_cache, time):
    import os
    import sys
 
    import torch
    from huggingface_hub import get_token
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
 
    # Importing fairchem loads NVIDIA Warp, whose compiled code prints a "Could not find or load the
    # NVIDIA CUDA driver" note straight to the OS-level stderr on machines without a GPU. That's
    # expected here (we fall back to the CPU), so mute stderr just for this import.
    sys.stderr.flush()
    _saved_fd = os.dup(2)
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(_null_fd, 2)
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    finally:
        os.dup2(_saved_fd, 2)
        os.close(_null_fd)
        os.close(_saved_fd)
 
    MODEL_NAME = "uma-s-1p2p1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Use the token from the box; otherwise fall back to an HF_TOKEN environment variable
    # or a token saved by `hf auth login`.
    _token = hf_token.value.strip() or get_token()
    mo.stop(
        device not in predictor_cache and not _token,
        mo.callout(mo.md("⬆️ **Paste your Hugging Face token above** to download UMA. "
                         "Everything below needs the model."), kind="warn"),
    )
    if hf_token.value.strip():
        os.environ["HF_TOKEN"] = hf_token.value.strip()
 
    _t0 = time.perf_counter()
    if device not in predictor_cache:
        try:
            with mo.status.spinner(title=f"Loading {MODEL_NAME} (first use downloads the weights)…"):
                # "batch" mode keeps the full model, which suits a notebook that keeps switching between
                # different molecules and crystals (the default mode specializes to one composition).
                predictor_cache[device] = pretrained_mlip.get_predict_unit(
                    MODEL_NAME, inference_settings="batch", device=device
                )
        except GatedRepoError:
            mo.stop(True, mo.callout(mo.md(
                "🔒 **Your token works, but this account doesn't have access to UMA yet.** Request access at "
                "[huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA) and wait for the approval "
                "email, then re-run this cell."), kind="danger"))
        except HfHubHTTPError as _err:
            mo.stop(True, mo.callout(mo.md(
                "❌ **Hugging Face refused the download.** Check that you pasted the whole token (it starts with "
                f"`hf_`) and that it hasn't been deleted.\n\n`{type(_err).__name__}: {str(_err).splitlines()[0]}`"),
                kind="danger"))
    predictor = predictor_cache[device]
 
    _calcs = {
        "omol": FAIRChemCalculator(predictor, task_name="omol"),
        "omat": FAIRChemCalculator(predictor, task_name="omat"),
        "oc20": FAIRChemCalculator(predictor, task_name="oc20"),
    }
 
    def calc_for(atoms, task=None):
        """Pick the UMA task head. Default: `omat` for periodic crystals, `omol` for isolated molecules."""
        if task is None:
            task = "omat" if atoms.pbc.any() else "omol"
        if task == "omol":
            atoms.info.setdefault("charge", 0)  # neutral
            atoms.info.setdefault("spin", 1)    # spin multiplicity 2S+1 (1 = singlet)
        return _calcs[task]
 
    mo.callout(
        mo.md(
            f"✅ **{MODEL_NAME}** is ready on **{device.upper()}** "
            f"({time.perf_counter() - _t0:.1f} s)."
        ),
        kind="success",
    )
    return (calc_for,)
 
 
@app.cell(hide_code=True)
def _(FrechetCellFilter, GPa, calc_for, np, time):
    def energy_and_forces(atoms, task=None):
        """Single-point evaluation: energy (eV) and forces (eV/Å) for a copy of `atoms`."""
        a = atoms.copy()
        a.calc = calc_for(a, task)
        return a.get_potential_energy(), a.get_forces()
 
    def max_force(forces):
        """The largest force on any atom: the number optimizers compare against `fmax`."""
        return float(np.sqrt((np.asarray(forces) ** 2).sum(axis=1)).max())
 
    def run_optimization(atoms, optimizer_cls, fmax, steps, use_cell_filter=False, task=None, **opt_kwargs):
        """Relax `atoms` in place and record a snapshot at every step."""
        atoms.calc = calc_for(atoms, task)
        target = FrechetCellFilter(atoms) if use_cell_filter else atoms
        want_pressure = bool(atoms.pbc.all()) and task in (None, "omat")
        frames = []
 
        def record():
            frames.append(
                {
                    "atoms": atoms.copy(),
                    "energy": atoms.get_potential_energy(),
                    "forces": atoms.get_forces().copy(),
                    # For a cell filter this includes the stress "forces" on the cell.
                    "fmax": max_force(target.get_forces()),
                    "pressure": -atoms.get_stress(voigt=True)[:3].mean() / GPa if want_pressure else None,
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
 
    def structure_figure(atoms, forces=None, force_scale=0.6, ranges=None, title=None, height=420,
                         show_cell=True, bond_atoms=None):
        """Interactive ball-and-stick 3D view of `atoms` (with optional force arrows) using plotly.
 
        `bond_atoms`: if given, only draw bonds that touch one of these atom indices
        (handy for a slab, where metal-metal "bonds" would just clutter the picture).
        """
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
        keep = None if bond_atoms is None else set(int(i) for i in bond_atoms)
        for a, b, off in bonds_of(atoms):
            if keep is not None and a not in keep and b not in keep:
                continue
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
        if show_cell and atoms.pbc.any():
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
 
    r_grid = np.linspace(0.7 * r_ref, 1.55 * r_ref, 90)
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
        value=round(1.35 * r_ref, 2), label="Starting bond length $r_0$ (Å)", show_value=True, debounce=True,
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
                r_new = float(np.clip(r + alpha.value * f, 0.5 * r_grid[0], 1.03 * r_grid[-1]))
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
    _fig.add_trace(go.Scatter(x=r_grid, y=e_grid, mode="lines", name="UMA energy",
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
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")  # 3. attach UMA (molecule head)
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
                      "bonds won't match it perfectly: the reference (MP2) and UMA's `omol` head (ωB97M-V) use different "
                      "levels of theory, so differences of ~0.01 Å are expected."),
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
    atoms.calc = FAIRChemCalculator(predictor, task_name="omat")   # crystal head
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
    ## 5 · Catalysis: where does a molecule land? 🎯
 
    Much of the chemical industry runs on **heterogeneous catalysis**: ammonia for fertilizer (on iron or
    ruthenium), the catalytic converter in your car (Pt, Pd, and Rh turning CO into CO₂), hydrogen fuel cells
    (Pt), CO₂ recycling (Cu). Every one of these starts the same way: a molecule **sticks to a metal surface**.
    *Where* it sits and *how strongly* it binds decide what happens next. Bind too weakly and it never reacts;
    bind too strongly and it never leaves (the **Sabatier principle**).
 
    We model a surface as a **slab**: a few atomic layers, repeated forever sideways, with vacuum above.
    The bottom layers are frozen at their bulk positions. On a close-packed fcc(111) surface an adsorbate
    can sit in four high-symmetry **sites**:
 
    | Site | Sits above | Metal neighbours |
    |---|---|---|
    | **top** | one surface atom | 1 |
    | **bridge** | the midpoint between two surface atoms | 2 |
    | **fcc hollow** | a triangle of surface atoms, with *no* atom directly below in the 2nd layer | 3 |
    | **hcp hollow** | a triangle of surface atoms, with a 2nd-layer atom directly below | 3 |
 
    (The square fcc(100) surface has top, bridge, and a 4-fold hollow.)
 
    ```python
    from ase import Atoms
    from ase.build import fcc111, add_adsorbate
    from ase.constraints import FixAtoms
 
    slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0, periodic=True)
    slab.set_constraint(FixAtoms(mask=slab.get_tags() > 1))   # freeze all but the top layer (tag 1)
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]])     # carbon end down
    add_adsorbate(slab, co, height=1.8, position="ontop")
    slab.calc = FAIRChemCalculator(predictor, task_name="oc20")   # the catalysis head
    BFGS(slab).run(fmax=0.03)
    ```
 
    **How strongly does it stick?** We report the adsorption energy in the convention of the
    [Open Catalyst 2020](https://arxiv.org/abs/2010.09990) dataset that UMA's `oc20` head learned from:
 
    $$
    E_\text{ads} = E(\text{slab + adsorbate}) - E(\text{clean slab}) - \sum_i n_i\,\varepsilon_i
    $$
 
    The atomic references $\varepsilon_i$ come from gas molecules: $\varepsilon_\text{H} = \tfrac12 E(\text{H}_2)$,
    $\varepsilon_\text{O} = E(\text{H}_2\text{O}) - E(\text{H}_2)$, $\varepsilon_\text{C} = E(\text{CO}) - \varepsilon_\text{O}$,
    $\varepsilon_\text{N} = \tfrac12 E(\text{N}_2)$. So for O the number is the energy of
    H₂O(g) + surface → O\* + H₂(g). **More negative means stickier.**
    """)
    return
 
 
@app.cell(hide_code=True)
def _(Atoms, mo):
    # Experimental lattice constants (Å). Good enough for a demo; for research you'd use the
    # lattice constant of the same DFT method the model learned (RPBE for oc20).
    surface_metals = {
        "Platinum · Pt": ("Pt", 3.924),
        "Palladium · Pd": ("Pd", 3.891),
        "Rhodium · Rh": ("Rh", 3.803),
        "Iridium · Ir": ("Ir", 3.839),
        "Nickel · Ni": ("Ni", 3.524),
        "Copper · Cu": ("Cu", 3.615),
        "Silver · Ag": ("Ag", 4.086),
        "Gold · Au": ("Au", 4.078),
    }
    # (adsorbate with the binding atom first and the molecule standing upright, starting height in Å)
    adsorbate_choices = {
        "H · hydrogen atom": (Atoms("H"), 1.0),
        "O · oxygen atom": (Atoms("O"), 1.2),
        "N · nitrogen atom": (Atoms("N"), 1.1),
        "OH · hydroxyl": (Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.98]]), 1.4),
        "CO · carbon monoxide": (Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]]), 1.4),
        "NO · nitric oxide": (Atoms("NO", positions=[[0, 0, 0], [0, 0, 1.17]]), 1.3),
    }
    metal_picker = mo.ui.dropdown(options=list(surface_metals), value="Platinum · Pt", label="**Metal**")
    facet_picker = mo.ui.radio(options={"fcc(111) · close-packed": "111", "fcc(100) · square": "100"},
                               value="fcc(111) · close-packed", label="**Surface**", inline=True)
    ads_picker = mo.ui.dropdown(options=list(adsorbate_choices), value="CO · carbon monoxide",
                                label="**Adsorbate**")
    nudge_switch = mo.ui.switch(value=False, label="Nudge each start sideways by ~0.1 Å")
    mo.hstack([mo.vstack([metal_picker, ads_picker]), mo.vstack([facet_picker, nudge_switch])], widths="equal")
    return (
        adsorbate_choices,
        ads_picker,
        facet_picker,
        metal_picker,
        nudge_switch,
        surface_metals,
    )
 
 
@app.cell(hide_code=True)
def _(add_adsorbate, find_mic, go, jmol_colors, np):
    # OC20 atomic reference energies (eV), from the OC20 paper's SI (also used by fairchem's AdsorbML recipe)
    OC20_REFS = {"H": -3.477, "O": -7.204, "C": -7.282, "N": -8.083}
    SITE_LABELS = {"ontop": "top", "bridge": "bridge", "fcc": "fcc hollow", "hcp": "hcp hollow",
                   "hollow": "4-fold hollow"}
    SITE_COLORS = {"top": "#e4572e", "bridge": "#f58518", "fcc hollow": "#4c78a8", "hcp hollow": "#54a24b",
                   "4-fold hollow": "#4c78a8", "in between": "#9d9d9d", "desorbed": "#b279a2",
                   "dissociated": "#000000"}
    SUPER = 3  # the slab is a 3×3 repeat of the primitive surface cell
 
    def prim_cell(slab):
        """2×2 matrix whose rows are the primitive surface lattice vectors (x, y)."""
        return np.asarray(slab.info["adsorbate_info"]["cell"])
 
    def drop_xy(slab, u, v):
        """Cartesian (x, y) of fractional point (u, v) inside the *central* primitive cell."""
        return np.array([1 + u, 1 + v]) @ prim_cell(slab)
 
    def place_adsorbate(slab, ads, height, position):
        """Copy `slab` and add `ads` (binding atom first) at a site name or an (x, y) position."""
        s = slab.copy()
        if isinstance(position, str):
            add_adsorbate(s, ads.copy(), height, position=position, offset=(1, 1))
        else:
            add_adsorbate(s, ads.copy(), height, position=tuple(position))
        return s
 
    def adsorption_energy(e_system, e_slab, ads):
        return e_system - e_slab - sum(OC20_REFS[s] for s in ads.get_chemical_symbols())
 
    def _lateral(atoms, idx, point):
        d = atoms.positions[idx] - point
        d[:, 2] = 0.0
        d, _ = find_mic(d, atoms.cell, pbc=[True, True, False])
        return np.sort(np.linalg.norm(d[:, :2], axis=1))
 
    def classify_site(atoms, n_ads, facet):
        """Name the site the adsorbate's binding atom ended up on."""
        n_slab = len(atoms) - n_ads
        tags = atoms.get_tags()[:n_slab]
        top = np.where(tags == 1)[0]
        anchor = atoms.positions[n_slab]
        a_nn = np.linalg.norm(prim_cell(atoms)[0])
        if anchor[2] - atoms.positions[top, 2].max() > 3.2:
            return "desorbed"
        if n_ads > 1 and atoms.get_distance(n_slab, n_slab + 1, mic=True) > 1.6:
            return "dissociated"
        lat = _lateral(atoms, top, anchor)
        tol = 0.12 * a_nn
        if lat[0] < 0.25 * a_nn:
            return "top"
        n_hollow = 3 if facet == "111" else 4
        if lat[n_hollow - 1] - lat[0] < tol:
            if facet == "100":
                return "4-fold hollow"
            sub = _lateral(atoms, np.where(tags == 2)[0], anchor)
            return "hcp hollow" if sub[0] < 0.3 * a_nn else "fcc hollow"
        if lat[1] - lat[0] < tol:
            return "bridge"
        return "in between"
 
    def _interp_periodic(E, u, v):
        """Bilinear interpolation of an n×n periodic grid E at fractional coords (u, v)."""
        n = E.shape[0]
        x, y = (u % 1.0) * n, (v % 1.0) * n
        i0, j0 = np.floor(x).astype(int) % n, np.floor(y).astype(int) % n
        tx, ty = x - np.floor(x), y - np.floor(y)
        i1, j1 = (i0 + 1) % n, (j0 + 1) % n
        return ((1 - tx) * (1 - ty) * E[i0, j0] + tx * (1 - ty) * E[i1, j0]
                + (1 - tx) * ty * E[i0, j1] + tx * ty * E[i1, j1])
 
    def surface_map_figure(slab, heat=None, paths=(), drop=None, show_sites=True, height=470):
        """Top-down view of the surface. `heat` = n×n grid of energies over the primitive cell."""
        P = prim_cell(slab)
        S = SUPER * P
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ S
        _zone = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) + 1) @ P
        lo, hi = _zone.min(0) - 0.9 * np.linalg.norm(P[0]), _zone.max(0) + 0.9 * np.linalg.norm(P[0])
        fig = go.Figure()
 
        if heat is not None:
            xs = np.linspace(lo[0], hi[0], 160)
            ys = np.linspace(lo[1], hi[1], 140)
            X, Y = np.meshgrid(xs, ys)
            frac = np.stack([X.ravel(), Y.ravel()], 1) @ np.linalg.inv(P)
            Z = _interp_periodic(heat, frac[:, 0], frac[:, 1]).reshape(X.shape)
            fig.add_trace(go.Contour(
                x=xs, y=ys, z=Z, colorscale="Viridis", ncontours=22, line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
                colorbar=dict(title=dict(text="E<sub>ads</sub> (eV)", side="right"), thickness=14),
                hovertemplate="x=%{x:.2f} y=%{y:.2f}<br>E_ads=%{z:.2f} eV<extra></extra>",
            ))
 
        # Surface atoms (top two layers) as filled circles in real units (Å), including periodic images.
        # Drawn as polygon traces so they sit above the energy map but below the trails.
        tags = slab.get_tags()
        a_nn = np.linalg.norm(P[0])
        ang = np.linspace(0, 2 * np.pi, 41)
        for layer, r_frac, shade, alpha in ((2, 0.30, 0.55, 0.9), (1, 0.43, 1.0, 0.9 if heat is None else 0.45)):
            xs_, ys_ = [], []
            for i in np.where(tags == layer)[0]:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        x, y = slab.positions[i, :2] + di * S[0] + dj * S[1]
                        r = r_frac * a_nn
                        if x + r < lo[0] or x - r > hi[0] or y + r < lo[1] or y - r > hi[1]:
                            continue
                        xs_ += list(x + r * np.cos(ang)) + [None]
                        ys_ += list(y + r * np.sin(ang)) + [None]
            rgb = jmol_colors[slab.numbers[0]] * 255 * shade
            fig.add_trace(go.Scatter(
                x=xs_, y=ys_, mode="lines", fill="toself", hoverinfo="skip", showlegend=False,
                fillcolor=f"rgba({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f},{alpha})",
                line=dict(color="rgba(40,40,40,0.7)", width=1),
            ))
 
        # Slab outline and the "drop zone" (central primitive cell)
        fig.add_trace(go.Scatter(x=corners[:, 0], y=corners[:, 1], mode="lines", hoverinfo="skip",
                                 line=dict(color="rgba(90,120,200,0.8)", width=2), showlegend=False))
        dz = (np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) + 1) @ P
        fig.add_trace(go.Scatter(x=dz[:, 0], y=dz[:, 1], mode="lines", hoverinfo="skip", name="drop zone",
                                 line=dict(color="#e4572e", width=2, dash="dash"), showlegend=False))
 
        if show_sites:
            for key, (fu, fv) in slab.info["adsorbate_info"]["sites"].items():
                xy = np.array([1 + fu, 1 + fv]) @ P
                fig.add_trace(go.Scatter(
                    x=[xy[0]], y=[xy[1]], mode="markers+text", text=[SITE_LABELS[key]], textposition="top center",
                    textfont=dict(size=13, color="black"), hoverinfo="skip", showlegend=False,
                    marker=dict(size=10, color=SITE_COLORS[SITE_LABELS[key]], line=dict(color="white", width=1.5)),
                ))
 
        seen = set()
        for p in paths:
            xy, ended = np.asarray(p["path"]), p["ended"]
            col = SITE_COLORS.get(ended, "#444")
            fig.add_trace(go.Scatter(
                x=xy[:, 0], y=xy[:, 1], mode="lines", line=dict(color=col, width=2.5),
                name=ended, legendgroup=ended, showlegend=ended not in seen, hoverinfo="skip",
            ))
            seen.add(ended)
            fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", legendgroup=ended,
                                     marker=dict(symbol="x", size=9, color=col), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[xy[-1, 0]], y=[xy[-1, 1]], mode="markers", legendgroup=ended, showlegend=False,
                marker=dict(size=12, color=col, line=dict(color="white", width=2)),
                hovertemplate=f"ended on: {ended}<br>E_ads = {p['E_ads']:.2f} eV<extra></extra>",
            ))
 
        if drop is not None:
            fig.add_trace(go.Scatter(x=[drop[0]], y=[drop[1]], mode="markers", name="drop here",
                                     marker=dict(symbol="star", size=20, color="#ffd400",
                                                 line=dict(color="black", width=1.5)),
                                     hoverinfo="skip", showlegend=False))
 
        fig.update_layout(
            height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, range=[lo[0], hi[0]], constrain="domain"),
            yaxis=dict(visible=False, range=[lo[1], hi[1]], scaleanchor="x", scaleratio=1, constrain="domain"),
            legend=dict(orientation="h", y=-0.02, title_text="ended on: "),
        )
        return fig
 
    return (
        OC20_REFS,
        SITE_LABELS,
        adsorption_energy,
        classify_site,
        drop_xy,
        place_adsorbate,
        surface_map_figure,
    )
 
 
@app.cell(hide_code=True)
def _(
    BFGS,
    FixAtoms,
    adsorbate_choices,
    ads_picker,
    facet_picker,
    fcc100,
    fcc111,
    metal_picker,
    mo,
    run_optimization,
    surface_metals,
):
    surf_metal, _a = surface_metals[metal_picker.value]
    surf_facet = facet_picker.value
    ads_atoms, ads_height = adsorbate_choices[ads_picker.value]
    n_ads = len(ads_atoms)
 
    _builder = fcc111 if surf_facet == "111" else fcc100
    _slab = _builder(surf_metal, size=(3, 3, 3), a=_a, vacuum=10.0, periodic=True)
    _slab.set_constraint(FixAtoms(mask=_slab.get_tags() > 1))  # tag 1 = top layer
    with mo.status.spinner(title=f"Relaxing the clean {surf_metal}({surf_facet}) surface…"):
        _res = run_optimization(_slab, BFGS, fmax=0.03, steps=100, task="oc20")
    clean_slab = _res["frames"][-1]["atoms"]
    E_slab = _res["frames"][-1]["energy"]
    return E_slab, ads_atoms, ads_height, clean_slab, n_ads, surf_facet, surf_metal
 
 
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5a · Site showdown 🥊
 
    We put the adsorbate on **every high-symmetry site**, relax each one, and compare. The lowest
    adsorption energy wins. Watch the *ended on* column: if an adsorbate wanders off to a different site
    during the relaxation, its starting site wasn't a minimum at all.
    """)
    return
 
 
@app.cell(hide_code=True)
def _(
    BFGS,
    E_slab,
    SITE_LABELS,
    adsorption_energy,
    ads_atoms,
    ads_height,
    classify_site,
    clean_slab,
    mo,
    n_ads,
    np,
    nudge_switch,
    place_adsorbate,
    run_optimization,
    surf_facet,
):
    _rng = np.random.default_rng(0)
    showdown = []
    _sites = list(clean_slab.info["adsorbate_info"]["sites"])
    for _site in mo.status.progress_bar(_sites, title="Landing on every site…", remove_on_exit=True):
        _extra = {"ontop": 0.5, "bridge": 0.25}.get(_site, 0.0)  # top sites bind further out
        _start = place_adsorbate(clean_slab, ads_atoms, ads_height + _extra, _site)
        if nudge_switch.value:
            _start.positions[-n_ads:, :2] += _rng.normal(0, 0.1, 2)  # shift the whole adsorbate sideways
        _res = run_optimization(_start, BFGS, fmax=0.03, steps=250, task="oc20")
        _final = _res["frames"][-1]["atoms"]
        showdown.append({
            "site": SITE_LABELS[_site],
            "result": _res,
            "final": _final,
            "ended": classify_site(_final, n_ads, surf_facet),
            "E_ads": adsorption_energy(_res["frames"][-1]["energy"], E_slab, ads_atoms),
            "height": _final.positions[-n_ads, 2] - _final.positions[:-n_ads, 2].max(),
        })
    return (showdown,)
 
 
@app.cell(hide_code=True)
def _(mo, showdown):
    _best = min(showdown, key=lambda r: r["E_ads"])["site"]
    showdown_view = mo.ui.dropdown(options=[r["site"] for r in showdown], value=_best,
                                   label="**Show the relaxed structure that started on**")
    return (showdown_view,)
 
 
@app.cell(hide_code=True)
def _(
    ads_picker,
    go,
    mo,
    n_ads,
    np,
    nudge_switch,
    showdown,
    showdown_view,
    structure_figure,
    surf_facet,
    surf_metal,
):
    _SC = {"top": "#e4572e", "bridge": "#f58518", "fcc hollow": "#4c78a8", "hcp hollow": "#54a24b",
           "4-fold hollow": "#4c78a8"}
    _best = min(showdown, key=lambda r: r["E_ads"])
    _ads_name = ads_picker.value.split(" · ")[0]
 
    _bar = go.Figure(go.Bar(
        x=[r["site"] for r in showdown], y=[r["E_ads"] for r in showdown],
        marker_color=[_SC.get(r["site"], "#888") for r in showdown],
        text=[f"{r['E_ads']:.2f} eV" + ("" if r["ended"] == r["site"] else f"<br>→ {r['ended']}")
              for r in showdown],
        textposition="outside",
        hovertemplate="started on %{x}<br>E_ads = %{y:.3f} eV<extra></extra>",
    ))
    _ys = [r["E_ads"] for r in showdown]
    _pad = 0.25 * (max(_ys) - min(_ys) + 0.2)
    _bar.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="adsorption energy (eV)  ← stickier", xaxis_title="starting site",
        yaxis_range=[min(min(_ys), 0) - _pad, max(max(_ys), 0) + _pad],
    )
 
    _sel = next(r for r in showdown if r["site"] == showdown_view.value)
    _fr = _sel["result"]["frames"][-1]
    _n = len(_sel["final"])
    _view = structure_figure(_sel["final"], forces=None, height=380, show_cell=False,
                             bond_atoms=range(_n - n_ads, _n),
                             title=f"{_ads_name} on {surf_metal}({surf_facet}): started {_sel['site']}, "
                                   f"ended {_sel['ended']}")
 
    _table = [{
        "": "🏆" if r is _best else "",
        "started on": r["site"],
        "ended on": r["ended"] + ("" if r["ended"] == r["site"] else "  🎢"),
        "E_ads (eV)": round(r["E_ads"], 3),
        "vs best (meV)": round(1000 * (r["E_ads"] - _best["E_ads"])),
        "height (Å)": round(r["height"], 2),
        "steps": len(r["result"]["frames"]) - 1,
    } for r in showdown]
 
    _notes = []
    _moved = [r for r in showdown if r["ended"] != r["site"]]
    if _moved:
        _notes.append(mo.callout(mo.md(
            "🎢 **Some adsorbates rolled away.** " + "; ".join(f"**{r['site']}** → {r['ended']}" for r in _moved)
            + ". Those starting sites are not minima for this adsorbate, so the optimizer slid downhill to "
              "a neighbouring site."), kind="warn"))
    elif not nudge_switch.value:
        _notes.append(mo.callout(mo.md(
            "🕵️ **Every adsorbate stayed exactly where we put it. Suspicious!** At a high-symmetry site the "
            "sideways forces cancel *exactly*, by symmetry (just like the strained crystal in section 4). So an "
            "optimizer can't tell a true valley from a **saddle point**: a mountain pass that is downhill "
            "sideways. Turn on **Nudge** above and see which sites survive."), kind="info"))
    else:
        _notes.append(mo.callout(mo.md(
            "✅ Every site survived the nudge, so each one is a genuine local minimum for this adsorbate."),
            kind="success"))
 
    # Hopping barrier estimate: a bridge that stayed a bridge is (roughly) the pass between two hollows
    _bridge = next((r for r in showdown if r["site"] == "bridge" and r["ended"] == "bridge"), None)
    _hollows = [r for r in showdown if "hollow" in r["ended"] and r["ended"] == r["site"]]
    _stats = [
        mo.stat(f"{_best['site']}", label="favourite site", bordered=True),
        mo.stat(f"{_best['E_ads']:.2f} eV", label="adsorption energy", caption="OC20 convention", bordered=True),
    ]
    if _bridge and _hollows:
        _ea = _bridge["E_ads"] - min(r["E_ads"] for r in _hollows)
        if _ea > 0.005:
            _rate = 1e13 * np.exp(-_ea / (8.617e-5 * 300))
            _stats.append(mo.stat(f"≈ {_ea:.2f} eV", label="hopping barrier (bridge − hollow)", bordered=True))
            _stats.append(mo.stat(f"{_rate:.1e} /s", label="hops per second at 300 K",
                                  caption="ν = 10¹³ s⁻¹ · exp(−Eₐ/kT)", bordered=True))
 
    if surf_metal == "Pt" and surf_facet == "111" and _ads_name == "CO":
        _verdict = ("UMA agrees with experiment 🎉" if _best["site"] == "top"
                    else "UMA sides with GGA-level DFT, not with experiment 🤔")
        _notes.append(mo.callout(mo.md(
            f"🧩 **The CO/Pt(111) puzzle.** Experiments show that, at low coverage, CO sits on the **top** site of "
            "Pt(111). Yet standard DFT functionals of the GGA family (PBE, and to a lesser degree RPBE, which "
            "the OC20 data used) famously prefer the **hollow** sites: a classic, much-studied failure "
            f"(Feibelman *et al.*, *J. Phys. Chem. B* 2001). Here UMA picks **{_best['site']}**: {_verdict}. "
            "A machine-learned potential can only be as right as the data it learned from."), kind="neutral"))
 
    mo.vstack([
        mo.hstack([_bar, mo.vstack([showdown_view, _view])], widths="equal"),
        mo.hstack(_stats, widths="equal"),
        mo.ui.table(_table, selection=None, pagination=False),
        *_notes,
    ])
    return
 
 
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5b · Drop it anywhere 🪂
 
    Real molecules don't land politely on a high-symmetry site. Pick a spot in the red **drop zone**,
    **predict** where the adsorbate will end up, then let go and watch the optimizer roll it downhill.
    Or press **Make it rain** to drop ten at random: the coloured trails sketch out the
    **basins of attraction**, the regions that drain into each site.
    """)
    return
 
 
@app.cell(hide_code=True)
def _(mo):
    get_drop_paths, set_drop_paths = mo.state([])
    return get_drop_paths, set_drop_paths
 
 
@app.cell(hide_code=True)
def _(ads_picker, clean_slab, set_drop_paths):
    # A new surface or adsorbate wipes the old trails.
    _ = (clean_slab, ads_picker.value)
    set_drop_paths([])
    return
 
 
@app.cell(hide_code=True)
def _(SITE_LABELS, clean_slab, mo, set_drop_paths):
    drop_u = mo.ui.slider(0.0, 1.0, step=0.02, value=0.16, label="Drop position along **a₁**",
                          show_value=True, full_width=True)
    drop_v = mo.ui.slider(0.0, 1.0, step=0.02, value=0.52, label="Drop position along **a₂**",
                          show_value=True, full_width=True)
    prediction = mo.ui.radio(
        options=[SITE_LABELS[s] for s in clean_slab.info["adsorbate_info"]["sites"]]
        + ["it stays where I dropped it"],
        label="**Your prediction: it will end up on…**",
    )
    let_go = mo.ui.run_button(label="🪂 Let go!", kind="success")
    rain = mo.ui.run_button(label="🌧️ Make it rain (10 random drops)")
    clear_drops = mo.ui.button(label="🧹 Clear trails", on_click=lambda v: (set_drop_paths([]), (v or 0) + 1)[1])
    return clear_drops, drop_u, drop_v, let_go, prediction, rain
 
 
@app.cell(hide_code=True)
def _(
    clean_slab,
    clear_drops,
    drop_u,
    drop_v,
    drop_xy,
    get_drop_paths,
    let_go,
    mo,
    prediction,
    rain,
    surface_map_figure,
):
    _xy = drop_xy(clean_slab, drop_u.value, drop_v.value)
    _fig = surface_map_figure(clean_slab, paths=get_drop_paths(), drop=_xy)
    mo.hstack([
        mo.vstack([drop_u, drop_v, prediction, mo.hstack([let_go, rain, clear_drops], justify="start", wrap=True),
                   mo.md("<small>⭐ = drop point · ✕ = start of a trail · ● = where it settled. "
                         "Big circles are the top-layer atoms, small dark ones the second layer.</small>")]),
        _fig,
    ], widths=[2, 3], align="center")
    return
 
 
@app.cell(hide_code=True)
def _(
    BFGS,
    E_slab,
    adsorption_energy,
    ads_atoms,
    ads_height,
    classify_site,
    clean_slab,
    drop_u,
    drop_v,
    drop_xy,
    let_go,
    mo,
    n_ads,
    np,
    place_adsorbate,
    rain,
    run_optimization,
    set_drop_paths,
    surf_facet,
):
    mo.stop(not (let_go.value or rain.value),
            mo.md("*Make your prediction, then press **Let go!** (or **Make it rain**).*"))
    drop_single = bool(let_go.value)
    if drop_single:
        _uvs = [(drop_u.value, drop_v.value)]
    else:
        _uvs = np.random.default_rng().random((10, 2))
    _new = []
    for _u, _v in mo.status.progress_bar(list(_uvs), title="Dropping…", remove_on_exit=True):
        _start = place_adsorbate(clean_slab, ads_atoms, ads_height + 0.5, drop_xy(clean_slab, _u, _v))
        _res = run_optimization(_start, BFGS, fmax=0.03, steps=300, task="oc20")
        _final = _res["frames"][-1]["atoms"]
        _new.append({
            "path": np.array([f["atoms"].positions[len(clean_slab), :2] for f in _res["frames"]]),
            "ended": classify_site(_final, n_ads, surf_facet),
            "E_ads": adsorption_energy(_res["frames"][-1]["energy"], E_slab, ads_atoms),
            "result": _res,
        })
    set_drop_paths(lambda old: old + [{k: d[k] for k in ("path", "ended", "E_ads")} for d in _new])
    last_drop = _new[-1]
    rain_drops = _new
    return drop_single, last_drop, rain_drops
 
 
@app.cell(hide_code=True)
def _(last_drop, mo):
    drop_frame = mo.ui.slider(
        start=0, stop=len(last_drop["result"]["frames"]) - 1, step=1,
        value=len(last_drop["result"]["frames"]) - 1, label="🎞️ Step", show_value=True, full_width=True,
    )
    return (drop_frame,)
 
 
@app.cell(hide_code=True)
def _(
    clean_slab,
    drop_frame,
    drop_single,
    last_drop,
    mo,
    n_ads,
    np,
    prediction,
    rain_drops,
    structure_figure,
    trajectory_ranges,
):
    _frames = last_drop["result"]["frames"]
    _i = drop_frame.value
    _n = len(clean_slab) + n_ads
    _view = structure_figure(_frames[_i]["atoms"], forces=_frames[_i]["forces"], height=420, show_cell=False,
                             bond_atoms=range(_n - n_ads, _n), ranges=trajectory_ranges(_frames, pad=1.0),
                             title=f"step {_i} · E_ads = {_frames[_i]['energy'] - _frames[-1]['energy'] + last_drop['E_ads']:.2f} eV")
    _travel = float(np.linalg.norm(last_drop["path"][-1] - last_drop["path"][0]))
 
    if drop_single:
        _ended = last_drop["ended"]
        _guess = prediction.value
        _stayed = _travel < 0.15
        if _guess is None:
            _msg = mo.callout(mo.md(f"It rolled **{_travel:.2f} Å** and settled on **{_ended}**. "
                                    "(Next time, make a prediction first!)"), kind="info")
        elif (_guess == _ended) or (_guess.startswith("it stays") and _stayed):
            _msg = mo.callout(mo.md(f"🎯 **You called it!** It rolled {_travel:.2f} Å and settled on **{_ended}**."),
                              kind="success")
        else:
            _msg = mo.callout(mo.md(f"😮 **Surprise!** You said *{_guess}*, but it rolled {_travel:.2f} Å "
                                    f"and settled on **{_ended}**. Section 5c shows the landscape that explains why."),
                              kind="warn")
        _summary = _msg
    else:
        _counts = {}
        for _d in rain_drops:
            _counts[_d["ended"]] = _counts.get(_d["ended"], 0) + 1
        _summary = mo.callout(mo.md(
            "🌧️ **Where the rain ended up:** " + ", ".join(f"**{k}** × {v}" for k, v in
                                                        sorted(_counts.items(), key=lambda kv: -kv[1]))
            + ". The share of drops that drain into each site is a rough measure of how big its basin is. "
              "A big basin isn't the same as a deep one: compare with the energies in 5a."), kind="info")
 
    mo.vstack([
        _summary,
        mo.md("**Replay the last drop** (red arrows are the forces pushing it downhill):"),
        _view,
        drop_frame,
    ])
    return
 
 
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5c · Map the energy landscape 🗺️
 
    In section 1 we could *see* the energy curve. Can we do the same here? Yes, if we restrict ourselves to
    the two sideways coordinates of the adsorbate. On a grid of points across one surface cell we pin the
    binding atom's $x$ and $y$ (with `FixCartesian`), let it relax only up and down, and record the energy.
    The result is the **potential energy surface** the adsorbate glides over: valleys are adsorption
    sites, mountain passes are the routes it takes when it diffuses.
 
    To keep it fast the metal atoms are frozen here, so the numbers are a bit less negative than in 5a.
    Your drop trails from 5b are drawn on top: they should run downhill across the colours.
    """)
    return
 
 
@app.cell(hide_code=True)
def _(mo):
    map_res = mo.ui.dropdown(options={"6 × 6  (36 relaxations, quick)": 6, "8 × 8  (64 relaxations)": 8,
                                      "10 × 10  (100 relaxations, sharpest)": 10},
                             value="6 × 6  (36 relaxations, quick)", label="**Grid**")
    map_button = mo.ui.run_button(label="🗺️ Map it", kind="success")
    mo.hstack([map_res, map_button], justify="start", align="end")
    return map_button, map_res
 
 
@app.cell(hide_code=True)
def _(
    BFGS,
    E_slab,
    FixAtoms,
    FixCartesian,
    adsorption_energy,
    ads_atoms,
    ads_height,
    clean_slab,
    drop_xy,
    map_button,
    map_res,
    mo,
    np,
    place_adsorbate,
    run_optimization,
):
    mo.stop(not map_button.value, mo.md("*Press **Map it** to scan the surface (a GPU makes this much faster).*"))
    _n = map_res.value
    pes = np.zeros((_n, _n))
    _nslab = len(clean_slab)
    for _k in mo.status.progress_bar(range(_n * _n), title="Scanning the surface…", remove_on_exit=True):
        _i, _j = divmod(_k, _n)
        _s = place_adsorbate(clean_slab, ads_atoms, ads_height + 0.4, drop_xy(clean_slab, _i / _n, _j / _n))
        _s.set_constraint([FixAtoms(indices=range(_nslab)), FixCartesian([_nslab], mask=(True, True, False))])
        _r = run_optimization(_s, BFGS, fmax=0.05, steps=80, task="oc20")
        pes[_i, _j] = adsorption_energy(_r["frames"][-1]["energy"], E_slab, ads_atoms)
    return (pes,)
 
 
@app.cell(hide_code=True)
def _(clean_slab, get_drop_paths, mo, pes, surface_map_figure):
    _fig = surface_map_figure(clean_slab, heat=pes, paths=get_drop_paths(), height=520)
    _corr = float(pes.max() - pes.min())
    mo.vstack([
        _fig,
        mo.hstack([
            mo.stat(f"{pes.min():.2f} eV", label="deepest valley", bordered=True),
            mo.stat(f"{pes.max():.2f} eV", label="highest point", bordered=True),
            mo.stat(f"{_corr:.2f} eV", label="corrugation (max − min)",
                    caption="how bumpy the surface feels", bordered=True),
        ], widths="equal"),
        mo.callout(mo.md(
            "**Reading the map.** Dark = low energy = where the adsorbate wants to be. A small corrugation "
            "(≲ 0.1 eV, typical for H or CO on many metals) means the adsorbate skates around easily at room "
            "temperature; a large one (O, N) means it's pinned in place. The lowest *pass* between two valleys "
            "is the diffusion barrier, and comparing it with $k_BT \\approx 0.026$ eV tells you how mobile it is."),
            kind="info"),
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
    - On surfaces, a high-symmetry start can be a **saddle point** that symmetry hides from the optimizer.
      Comparing all sites (and nudging) finds the real minimum; the energy difference between sites tells you
      how easily an adsorbate **diffuses**.
    - A universal ML potential like **UMA** makes all of this fast enough to explore interactively. Pick the
      task head that matches your system (`omol` for molecules, `omat` for inorganic crystals, `oc20` for
      catalyst surfaces). For publication-quality numbers, it's common to refine the final structure with DFT.
 
    ### 🧪 Things to try
    1. Set the shake strength to **0.4 Å** on ethanol and check the bond table: did a bond break or the molecule change conformer?
    2. In the race, tighten `fmax` to **0.005** and see which optimizer struggles most near the end.
    3. Relax all six crystals in section 4 and compare with experiment. The `omat` head learned PBE, which
       tends to **overestimate** lattice constants by about 1%. Does UMA follow that trend?
    4. Compress **NaCl** by −8% and watch how hard it pushes back compared with **tungsten**.
    5. **Noble or not?** Run the site showdown for **O** on Au, Ag, Cu, Pt, Pd, Ni. Which metal holds oxygen most
       weakly? That's why gold doesn't tarnish. Plot E_ads against the metal's position in the periodic table.
    6. **Sabatier in action:** compare **O** and **OH** binding across the same metals. Fuel-cell catalysts
       need O to bind *just right*, and Pt sits near the top of the "volcano".
    7. Turn **Nudge** on for **H** on Pt(111) and read off the hopping barrier. How many hops per second at room
       temperature? Now try **O**: is it as mobile?
    8. Make it rain on **fcc(100)** vs **fcc(111)**. Which surface has more kinds of basins?
 
    ### 📋 Cheat sheet
    ```python
    from ase import Atoms
    from ase.build import molecule, bulk, fcc111, add_adsorbate
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter
    from ase.io import write
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    import torch
 
    # needs a Hugging Face token with UMA access: set HF_TOKEN or run `hf auth login`
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device=device)
 
    # Molecule: relax atomic positions
    mol = molecule("CH3CH2OH")
    mol.info.update(charge=0, spin=1)          # neutral singlet
    mol.calc = FAIRChemCalculator(predictor, task_name="omol")
    BFGS(mol, trajectory="ethanol.traj").run(fmax=0.02)   # view with: ase gui ethanol.traj
 
    # Crystal: relax positions + cell
    si = bulk("Si", "diamond", a=5.6, cubic=True)
    si.calc = FAIRChemCalculator(predictor, task_name="omat")
    BFGS(FrechetCellFilter(si)).run(fmax=0.01)
    print(si.cell.cellpar())
    write("si_relaxed.cif", si)
 
    # Surface + adsorbate: adsorption energy in the OC20 convention
    oc20 = FAIRChemCalculator(predictor, task_name="oc20")
    slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0, periodic=True)
    slab.set_constraint(FixAtoms(mask=slab.get_tags() > 1))
    slab.calc = oc20
    BFGS(slab).run(fmax=0.03)
    e_slab = slab.get_potential_energy()
 
    adslab = slab.copy()
    add_adsorbate(adslab, Atoms("O"), height=1.2, position="fcc")
    adslab.calc = oc20
    BFGS(adslab, trajectory="O_Pt111.traj").run(fmax=0.03)
    e_ads = adslab.get_potential_energy() - e_slab - (-7.204)   # ε_O from OC20
    print(f"E_ads(O on Pt(111), fcc) = {e_ads:.2f} eV")
    ```
 
    *UMA: [huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA) ·
    fairchem: [github.com/facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) ·
    OC20: [Chanussot et al., ACS Catal. 2021](https://arxiv.org/abs/2010.09990) ·
    ASE optimizers: [ASE documentation](https://wiki.fysik.dtu.dk/ase/ase/optimize.html)*
    """)
    return
 
 
if __name__ == "__main__":
    app.run()
 
Claude finished the response

This file type cannot be opened. (×2)
