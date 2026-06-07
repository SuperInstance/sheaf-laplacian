# sheaf-laplacian

> **The Laplacian, but sheaf-theoretic. Diffusion on cellular sheaves.**

[![crates.io](https://img.shields.io/crates/v/sheaf-laplacian.svg)](https://crates.io/crates/sheaf-laplacian)
[![docs.rs](https://docs.rs/sheaf-laplacian/badge.svg)](https://docs.rs/sheaf-laplacian)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust library for computing the sheaf Laplacian L₁ = δᵀδ on cellular sheaves defined over graphs. Implements coboundary operators, Hodge decomposition, sheaf diffusion, and consensus dynamics. Generalizes the graph Laplacian to settings where nodes carry different vector spaces and edges carry linear transformation maps.

---

## Table of Contents

- [What is a Sheaf Laplacian?](#what-is-a-sheaf-laplacian)
- [Why Does This Matter?](#why-does-this-matter)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Related Crates](#related-crates)
- [License](#license)

---

## What is a Sheaf Laplacian?

A **cellular sheaf** F on a graph G assigns:
- A **stalk** (vector space) F(v) to each node v
- A **restriction map** F(e): F(v) → F(w) to each edge e = (v, w)

The **sheaf Laplacian** L₀ = δᵀδ generalizes the ordinary graph Laplacian. Instead of comparing scalar values at neighboring nodes, it compares values after applying the edge restriction maps. This measures how far a signal (0-cochain) is from being a **global section** — a consistent assignment across all stalks.

```
Ordinary Laplacian:  (Lf)_i = Σ_j A_ij (f_i − f_j)
                                         (compare scalars)

Sheaf Laplacian:     (Lf)_i = Σ_j F_eᵀF_e f_i − Σ_j F_eᵀ f_j
                                         (compare after restriction maps)
```

The key insight: the graph Laplacian is just the sheaf Laplacian where every stalk is ℝ and every restriction map is the identity. Sheaves let you model **heterogeneous** multi-agent systems where each agent lives in a different state space.

```
Node 0: ℝ²  ──F₀₁──→  Node 1: ℝ³  ──F₁₂──→  Node 2: ℝ²
          2×3 matrix           3×2 matrix

Each node speaks a different "language" (vector space).
Restriction maps are "translators" between them.
```

## Why Does This Matter?

**For multi-agent consensus**: In ordinary consensus, agents share scalar opinions. In sheaf consensus, agents have different state spaces (positions, velocities, belief distributions) and still reach agreement through restriction maps. This is how heterogeneous robot teams coordinate.

**For opinion dynamics**: Hansen & Ghrist (2021) showed that opinion dynamics on discourse sheaves model how people with different conceptual frameworks can still communicate — the restriction maps encode shared vocabulary.

**For signal processing**: Sheaf diffusion provides a principled way to smooth signals that live on heterogeneous data — medical imaging where each pixel has different modalities, sensor networks with different measurement types.

**For distributed computing**: The Hodge decomposition splits any signal into harmonic (global sections), exact (gradient-like), and coexact (curl-like) components — revealing the topological structure of distributed data.

## Architecture

```
sheaf-laplacian
│
├── Matrix                     ← Dense matrix operations
│   ├── zeros(), identity()        Constructors
│   ├── matmul(), matvec()         Linear algebra
│   ├── transpose()                Transpose
│   ├── frobenius_norm(), trace()  Norms
│   └── is_symmetric()             Symmetry check
│
├── Sheaf / SheafEdge          ← Cellular sheaf on a graph
│   ├── new(stalk_dims)            Create with per-node stalk dimensions
│   ├── add_edge(src, tgt, map)    Add edge with restriction matrix
│   ├── add_identity_edge()        Edge with identity restriction
│   ├── coboundary()               δ: C⁰ → C¹ (coboundary operator)
│   ├── sheaf_laplacian()          L₀ = δᵀδ
│   └── connection_laplacian()     Normalized sheaf Laplacian
│
├── Hodge Decomposition        ← Signal decomposition
│   ├── hodge_decompose(signal)    Split into harmonic + exact + coexact
│   ├── harmonic_dimension()       dim ker(L) = H⁰(F)
│   └── exact/coexact components   Gradient-like and curl-like parts
│
└── SignalProcessing           ← Dynamics on sheaves
    ├── diffuse(signal, dt, steps) Heat diffusion on sheaf
    ├── consensus(signal, tol)     Iterate to global section
    └── dirichlet_energy(signal)   E(s) = sᵀLs
```

## Quick Start

```rust
use sheaf_laplacian::{
    Matrix, Sheaf,
    hodge_decompose,
    SignalProcessing,
};

// Create a sheaf on 3 nodes with stalk dimensions [2, 3, 2]
let mut sheaf = Sheaf::new(vec![2, 3, 2]);

// Add edges with restriction maps
// Node 0 (ℝ²) → Node 1 (ℝ³): a 3×2 matrix
let mut r01 = Matrix::zeros(3, 2);
r01.set(0, 0, 1.0); r01.set(1, 1, 1.0); r01.set(2, 0, 0.5);
sheaf.add_edge(0, 1, r01);

// Node 1 (ℝ³) → Node 2 (ℝ²): a 2×3 matrix
let mut r12 = Matrix::zeros(2, 3);
r12.set(0, 0, 1.0); r12.set(1, 1, 1.0);
sheaf.add_edge(1, 2, r12);

// Compute the sheaf Laplacian
let L = sheaf.sheaf_laplacian();
println!("Sheaf Laplacian: {}×{}", L.rows, L.cols);

// Decompose a signal using Hodge theory
let signal = vec![1.0, 0.5, 0.0, 1.0, 0.3, 0.8, 0.2];
let decomp = hodge_decompose(&sheaf, &signal);
println!("Dirichlet energy: {:.4}", decomp.dirichlet_energy);
println!("Harmonic dimension: {}", decomp.harmonic_dim);

// Diffuse a signal on the sheaf (heat equation)
let sp = SignalProcessing::new(&sheaf);
let diffused = sp.diffuse(&signal, 0.1, 50);
println!("After 50 diffusion steps: {:?}", diffused);

// Run consensus (converge to nearest global section)
let consensus = sp.consensus(&signal, 0.001, 1000);
println!("Consensus state: {:?}", consensus);
```

## API Reference

### Matrix

| Method | Returns | Description |
|--------|---------|-------------|
| `zeros(rows, cols)` | `Matrix` | Zero matrix |
| `identity(n)` | `Matrix` | Identity matrix |
| `get(i, j)` | `f64` | Access element |
| `set(i, j, v)` | `()` | Set element |
| `transpose()` | `Matrix` | Matrix transpose |
| `matmul(&other)` | `Matrix` | Matrix multiplication |
| `matvec(&v)` | `Vec<f64>` | Matrix-vector product |
| `frobenius_norm()` | `f64` | ‖A‖_F |
| `trace()` | `f64` | tr(A) |
| `is_symmetric(tol)` | `bool` | Symmetry check |

### Sheaf

| Method | Returns | Description |
|--------|---------|-------------|
| `new(stalk_dims)` | `Self` | Create sheaf with per-node dimensions |
| `add_edge(src, tgt, restriction)` | `()` | Edge with linear map |
| `add_identity_edge(src, tgt)` | `()` | Edge with identity map |
| `total_c0_dim()` | `usize` | Total 0-cochain dimension (Σ dim F(v)) |
| `total_c1_dim()` | `usize` | Total 1-cochain dimension |
| `coboundary()` | `Matrix` | δ: C⁰ → C¹ |
| `sheaf_laplacian()` | `Matrix` | L₀ = δᵀδ |
| `connection_laplacian()` | `Matrix` | Normalized L |

### Hodge Decomposition

| Function/Field | Returns | Description |
|----------------|---------|-------------|
| `hedge_decompose(&sheaf, signal)` | `HodgeDecomposition` | Full decomposition |
| `.harmonic` | `Vec<f64>` | Harmonic component |
| `.exact` | `Vec<f64>` | Exact (gradient) component |
| `.coexact` | `Vec<f64>` | Coexact (curl) component |
| `.dirichlet_energy` | `f64` | sᵀLs |
| `.harmonic_dim` | `usize` | dim ker(L) |

### SignalProcessing

| Method | Returns | Description |
|--------|---------|-------------|
| `new(&sheaf)` | `Self` | Create processor |
| `diffuse(signal, dt, steps)` | `Vec<f64>` | Heat diffusion |
| `consensus(signal, tol, max_steps)` | `Vec<f64>` | Converge to global section |
| `dirichlet_energy(signal)` | `f64` | Energy functional |

## Mathematical Background

### Cellular Sheaves

A cellular sheaf F on graph G = (V, E) assigns:
- **Stalks**: F(v) ∈ Vect for each v ∈ V
- **Restriction maps**: F_{v→w}: F(v) → F(w) for each edge (v,w) ∈ E

A **global section** (or sheaf section) is an assignment s ∈ ⊕_v F(v) such that for every edge (v,w):

```
F_{v→w}(s_v) = F_{w→v}(s_w)
```

### Coboundary Operator

The coboundary δ: C⁰(F) → C¹(F) maps 0-cochains to 1-cochains:

```
(δs)_{(v,w)} = F_{w→v}(s_w) − F_{v→w}(s_v)
```

This measures the "disagreement" across each edge after applying restriction maps.

### Sheaf Laplacian

```
L₀ = δᵀδ : C⁰(F) → C⁰(F)
```

Properties:
- L₀ is positive semidefinite
- ker(L₀) = global sections of F
- sᵀL₀s = ‖δs‖² = Dirichlet energy

### Hodge Decomposition

Any 0-cochain decomposes orthogonally:

```
s = s_harmonic + s_exact + s_coexact

s_harmonic ∈ ker(L₀)        (global sections)
s_exact ∈ im(δᵀ)             (gradient fields)
s_coexact ∈ im(δ)⊥ ∩ ker(δᵀ)⊥
```

### Sheaf Diffusion

Heat equation on the sheaf:

```
ds/dt = −L₀ s
```

Solution: s(t) = e^{-L₀t} s(0). As t → ∞, s(t) → the nearest global section (harmonic component of s(0)).

## Installation

```bash
cargo add sheaf-laplacian
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
sheaf-laplacian = "0.1"
```

## Related Crates

Part of the **SuperInstance Exocortex** math fleet:

- **[tropical-graph](https://github.com/SuperInstance/tropical-graph)** — Max-plus algebra on graphs
- **[graph-homology](https://github.com/SuperInstance/graph-homology)** — Clique complexes and Betti numbers
- **[cohomology-ring](https://github.com/SuperInstance/cohomology-ring)** — Cup products and cohomology operations
- **[persistent-agent](https://github.com/SuperInstance/persistent-agent)** — Topological fingerprints for agents
- **[categorical-coordination](https://github.com/SuperInstance/categorical-coordination)** — Category theory for coordination

## References

- Robinson, M. *Topological Signal Processing* (2014)
- Hansen, J. & Ghrist, R. *Opinion Dynamics on Discourse Sheaves* (2021)
- Curry, J. *Sheaves, Cosheaves and Applications* (2014)

## License

MIT © [SuperInstance](https://github.com/SuperInstance)

Part of the [Exocortex](https://github.com/SuperInstance/exocortex) project.
