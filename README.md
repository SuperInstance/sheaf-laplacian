# sheaf-laplacian

> **The Laplacian, but sheaf-theoretic. Diffusion on cellular sheaves.**

[![crates.io](https://img.shields.io/crates/v/sheaf-laplacian.svg)](https://crates.io/crates/sheaf-laplacian)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Computes the sheaf Laplacian L₁ = δᵀδ for cellular sheaves on graphs. Implements coboundary operators, Hodge decomposition, and sheaf diffusion for consensus and signal processing.

## What is a Sheaf Laplacian?

A **cellular sheaf** assigns vector spaces to nodes and linear maps to edges of a graph. The **sheaf Laplacian** generalizes the graph Laplacian — instead of comparing scalar values at neighboring nodes, it compares values after applying the edge restriction maps.

This enables:
- **Sheaf diffusion**: information flows respecting the sheaf structure
- **Hodge decomposition**: split signals into harmonic, exact, and coexact parts
- **Sheaf consensus**: agents with heterogeneous state spaces can still agree

## Quick Start

```rust
use sheaf_laplacian::{Sheaf, CoboundaryOperator};

// Define a sheaf on a graph
// Each node gets a vector space, each edge gets a linear map
// The coboundary operator δ: C⁰ → C¹ computes differences
// The sheaf Laplacian L = δᵀδ measures "non-flatness"
```

## Mathematical Background

For a sheaf F on graph G:
- **Stalk** F(v): vector space at node v
- **Restriction** F(v→w): linear map from F(v) to F(w)
- **Coboundary** δ: maps 0-cochains to 1-cochains
- **Sheaf Laplacian**: L₀ = δᵀδ — measures how far a 0-cochain is from being a global section

Hodge decomposition: any 0-cochain s = s_harmonic + s_exact + s_coexact

## References

- Robinson, M. *Topological Signal Processing* (2014)
- Hansen, J. & Ghrist, R. *Opinion Dynamics on Discourse Sheaves* (2021)

## License

MIT © [SuperInstance](https://github.com/SuperInstance)
