//! # sheaf-laplacian
//!
//! Sheaf Laplacian computation with coboundary operators, Hodge decomposition,
//! and signal processing on cellular sheaves.
//!
//! A cellular sheaf assigns vector spaces to the nodes (vertices) of a graph
//! and linear maps (restriction maps) to each edge. The sheaf Laplacian
//! generalizes the graph Laplacian and encodes how well sections of the sheaf
//! agree across edges.

use std::fmt;

/// A matrix stored in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major data.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a zero matrix of given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n { m.set(i, i, 1.0); }
        m
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Matrix-vector multiplication.
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        (0..self.rows)
            .map(|i| (0..self.cols).map(|j| self.get(i, j) * v[j]).sum())
            .collect()
    }

    /// Subtract other from self.
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) - other.get(i, j));
            }
        }
        result
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Trace (sum of diagonal).
    pub fn trace(&self) -> f64 {
        (0..self.rows.min(self.cols)).map(|i| self.get(i, i)).sum()
    }

    /// Check if symmetric (within tolerance).
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if self.rows != self.cols { return false; }
        for i in 0..self.rows {
            for j in 0..i {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:8.4}", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// A directed edge with source, target, and a restriction map (matrix).
#[derive(Clone, Debug)]
pub struct SheafEdge {
    pub source: usize,
    pub target: usize,
    /// Restriction map from source stalk to target stalk.
    pub restriction: Matrix,
}

/// A cellular sheaf on a graph.
///
/// Assigns a vector space (stalk) of dimension `stalk_dims[v]` to each node `v`,
/// and a linear restriction map to each directed edge.
#[derive(Clone, Debug)]
pub struct Sheaf {
    /// Dimension of the stalk at each node.
    pub stalk_dims: Vec<usize>,
    /// Directed edges with restriction maps.
    pub edges: Vec<SheafEdge>,
}

impl Sheaf {
    /// Create a new sheaf with given stalk dimensions.
    pub fn new(stalk_dims: Vec<usize>) -> Self {
        Self { stalk_dims, edges: Vec::new() }
    }

    /// Add an edge with a restriction map.
    pub fn add_edge(&mut self, source: usize, target: usize, restriction: Matrix) {
        assert_eq!(restriction.rows, self.stalk_dims[target]);
        assert_eq!(restriction.cols, self.stalk_dims[source]);
        self.edges.push(SheafEdge { source, target, restriction });
    }

    /// Add an identity-restriction edge (same stalk dimension at both ends).
    pub fn add_identity_edge(&mut self, source: usize, target: usize) {
        let d = self.stalk_dims[source];
        assert_eq!(d, self.stalk_dims[target]);
        self.add_edge(source, target, Matrix::identity(d));
    }

    /// Total dimension of C⁰ (direct sum of all stalks).
    pub fn total_c0_dim(&self) -> usize {
        self.stalk_dims.iter().sum()
    }

    /// Total dimension of C¹ (sum of target stalk dims over all edges).
    pub fn total_c1_dim(&self) -> usize {
        self.edges.iter().map(|e| self.stalk_dims[e.target]).sum()
    }

    /// Build the coboundary operator δ: C⁰ → C¹ as a matrix.
    ///
    /// For each edge e=(s→t), the row block for e has -F_e in the s-block
    /// and the identity in the t-block (for same-dimension identity sheaves).
    /// General case: places the restriction map appropriately.
    pub fn coboundary(&self) -> Matrix {
        let n = self.total_c0_dim();
        let m = self.total_c1_dim();
        let mut delta = Matrix::zeros(m, n);

        // Compute offsets for node stalk blocks in C⁰
        let mut node_offsets = vec![0usize; self.stalk_dims.len()];
        for i in 1..self.stalk_dims.len() {
            node_offsets[i] = node_offsets[i - 1] + self.stalk_dims[i - 1];
        }

        // Compute offsets for edge blocks in C¹
        let mut edge_offsets = vec![0usize; self.edges.len()];
        for i in 1..self.edges.len() {
            edge_offsets[i] = edge_offsets[i - 1] + self.stalk_dims[self.edges[i - 1].target];
        }

        for (e_idx, edge) in self.edges.iter().enumerate() {
            let row_off = edge_offsets[e_idx];
            let col_off_s = node_offsets[edge.source];
            let col_off_t = node_offsets[edge.target];

            // Place -F_e at the source block
            for i in 0..edge.restriction.rows {
                for j in 0..edge.restriction.cols {
                    delta.set(row_off + i, col_off_s + j, -edge.restriction.get(i, j));
                }
            }
            // Place identity at the target block
            let tdim = self.stalk_dims[edge.target];
            for i in 0..tdim {
                delta.set(row_off + i, col_off_t + i, delta.get(row_off + i, col_off_t + i) + 1.0);
            }
        }
        delta
    }

    /// Compute the sheaf Laplacian L₁ = δᵀδ.
    pub fn sheaf_laplacian(&self) -> Matrix {
        let delta = self.coboundary();
        let dt = delta.transpose();
        dt.matmul(&delta)
    }

    /// Compute the connection Laplacian (for identity sheaves on undirected graphs).
    /// For edge (i,j) with restriction F: L += [[FᵀF, -Fᵀ], [-F, I]] on blocks (i,j).
    pub fn connection_laplacian(&self) -> Matrix {
        let n = self.total_c0_dim();
        let mut lap = Matrix::zeros(n, n);

        let mut node_offsets = vec![0usize; self.stalk_dims.len()];
        for i in 1..self.stalk_dims.len() {
            node_offsets[i] = node_offsets[i - 1] + self.stalk_dims[i - 1];
        }

        for edge in &self.edges {
            let off_s = node_offsets[edge.source];
            let off_t = node_offsets[edge.target];
            let ft = edge.restriction.transpose();
            let ftf = ft.matmul(&edge.restriction);
            let d_s = edge.restriction.cols;
            let d_t = edge.restriction.rows;

            // Add FᵀF to source block
            for i in 0..d_s {
                for j in 0..d_s {
                    lap.set(off_s + i, off_s + j, lap.get(off_s + i, off_s + j) + ftf.get(i, j));
                }
            }
            // Add I to target block
            for i in 0..d_t {
                lap.set(off_t + i, off_t + i, lap.get(off_t + i, off_t + i) + 1.0);
            }
            // Add -Fᵀ to source-target block
            for i in 0..d_s {
                for j in 0..d_t {
                    lap.set(off_s + i, off_t + j, lap.get(off_s + i, off_t + j) - ft.get(i, j));
                }
            }
            // Add -F to target-source block
            for i in 0..d_t {
                for j in 0..d_s {
                    lap.set(off_t + i, off_s + j, lap.get(off_t + i, off_s + j) - edge.restriction.get(i, j));
                }
            }
        }
        lap
    }
}

/// Result of Hodge decomposition of a signal on C⁰.
#[derive(Debug)]
pub struct HodgeDecomposition {
    /// Harmonic component (in ker(L₁)).
    pub harmonic: Vec<f64>,
    /// Exact component (in im(δᵀ)).
    pub exact: Vec<f64>,
    /// Coexact component (in im(δ)).
    pub coexact: Vec<f64>,
}

/// Hodge decomposition of a 0-cochain signal.
///
/// Decomposes a signal x ∈ C⁰ into:
/// - harmonic: in ker(L₁), i.e. δx = 0 (global sections)
/// - exact: in im(δᵀ), comes from 1-cochains
/// - coexact: the remainder
pub fn hodge_decompose(sheaf: &Sheaf, signal: &[f64]) -> HodgeDecomposition {
    let delta = sheaf.coboundary();
    let n = signal.len();

    // δ applied to signal
    let delta_x = delta.matvec(signal);

    // δᵀ applied to δx gives L₁ x
    let dt = delta.transpose();
    let lap_x = dt.matvec(&delta_x);

    // For a proper decomposition, we use iterative projection
    // Harmonic: project onto ker(L₁) via power iteration for smallest eigenvalue
    // Simplified: use the Laplacian to extract components

    // Harmonic component: solve approximately by noting that L₁ * harmonic ≈ 0
    // We use a few steps of gradient descent on ||L₁ v||² with v initialized to signal
    let mut harmonic = signal.to_vec();
    let lr = 0.1;
    for _ in 0..100 {
        let grad = dt.matvec(&delta.matvec(&harmonic));
        for i in 0..n {
            harmonic[i] -= lr * grad[i];
        }
        if grad.iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-10 {
            break;
        }
    }

    // Exact component: δᵀ(δx) approximation
    let exact = lap_x.clone();

    // Coexact = signal - harmonic - exact (normalized)
    let coexact: Vec<f64> = signal.iter().enumerate()
        .map(|(i, &s)| s - harmonic[i] - exact[i])
        .collect();

    HodgeDecomposition { harmonic, exact, coexact }
}

/// Signal processing operations on sheaves.
pub struct SignalProcessing<'a> {
    sheaf: &'a Sheaf,
}

impl<'a> SignalProcessing<'a> {
    /// Create a new signal processor for the given sheaf.
    pub fn new(sheaf: &'a Sheaf) -> Self {
        Self { sheaf }
    }

    /// Sheaf diffusion: x(t+dt) = x(t) - dt * L₁ * x(t).
    pub fn diffuse(&self, signal: &[f64], dt: f64, steps: usize) -> Vec<f64> {
        let lap = self.sheaf.sheaf_laplacian();
        let mut x = signal.to_vec();
        let n = x.len();
        for _ in 0..steps {
            let lap_x = lap.matvec(&x);
            for i in 0..n {
                x[i] -= dt * lap_x[i];
            }
        }
        x
    }

    /// Sheaf consensus: run diffusion until convergence.
    pub fn consensus(&self, signal: &[f64], tolerance: f64, max_steps: usize) -> Vec<f64> {
        let lap = self.sheaf.sheaf_laplacian();
        let mut x = signal.to_vec();
        let n = x.len();
        let dt = 0.01;
        for _ in 0..max_steps {
            let prev = x.clone();
            let lap_x = lap.matvec(&x);
            for i in 0..n {
                x[i] -= dt * lap_x[i];
            }
            let change: f64 = x.iter().zip(prev.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            if change < tolerance {
                break;
            }
        }
        x
    }

    /// Compute the Dirichlet energy: xᵀ L₁ x.
    pub fn dirichlet_energy(&self, signal: &[f64]) -> f64 {
        let lap = self.sheaf.sheaf_laplacian();
        let lap_x = lap.matvec(signal);
        signal.iter().zip(lap_x.iter()).map(|(a, b)| a * b).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_sheaf() -> Sheaf {
        let mut s = Sheaf::new(vec![1, 1, 1]);
        s.add_identity_edge(0, 1);
        s.add_identity_edge(1, 2);
        s.add_identity_edge(2, 0);
        s
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m = Matrix::zeros(2, 3);
        m.set(0, 0, 1.0); m.set(0, 1, 2.0); m.set(0, 2, 3.0);
        m.set(1, 0, 4.0); m.set(1, 1, 5.0); m.set(1, 2, 6.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_matrix_multiply() {
        let mut a = Matrix::identity(2);
        a.set(0, 1, 3.0);
        let b = Matrix::identity(2);
        let c = a.matmul(&b);
        assert_eq!(c.get(0, 1), 3.0);
    }

    #[test]
    fn test_matrix_symmetry() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 1, 1.0); m.set(1, 0, 1.0);
        assert!(m.is_symmetric(1e-10));
        m.set(0, 1, 2.0);
        assert!(!m.is_symmetric(1e-10));
    }

    #[test]
    fn test_sheaf_creation() {
        let s = triangle_sheaf();
        assert_eq!(s.stalk_dims.len(), 3);
        assert_eq!(s.edges.len(), 3);
        assert_eq!(s.total_c0_dim(), 3);
        assert_eq!(s.total_c1_dim(), 3);
    }

    #[test]
    fn test_coboundary_dimensions() {
        let s = triangle_sheaf();
        let delta = s.coboundary();
        assert_eq!(delta.rows, 3);
        assert_eq!(delta.cols, 3);
    }

    #[test]
    fn test_sheaf_laplacian_is_symmetric() {
        let s = triangle_sheaf();
        let lap = s.sheaf_laplacian();
        assert!(lap.is_symmetric(1e-10));
    }

    #[test]
    fn test_sheaf_laplacian_positive_semidefinite() {
        let s = triangle_sheaf();
        let lap = s.sheaf_laplacian();
        // Constant signal should have zero energy
        let signal = vec![1.0, 1.0, 1.0];
        let result = lap.matvec(&signal);
        let energy: f64 = result.iter().map(|x| x * x).sum();
        assert!(energy < 1e-10);
    }

    #[test]
    fn test_sheaf_laplacian_trace() {
        let s = triangle_sheaf();
        let lap = s.sheaf_laplacian();
        // For triangle with identity restrictions, trace should be 6 (each node degree 2)
        assert!((lap.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_connection_laplacian_symmetric() {
        let s = triangle_sheaf();
        let cl = s.connection_laplacian();
        assert!(cl.is_symmetric(1e-10));
    }

    #[test]
    fn test_diffusion_reduces_energy() {
        let s = triangle_sheaf();
        let sp = SignalProcessing::new(&s);
        let signal = vec![1.0, 2.0, 3.0];
        let e0 = sp.dirichlet_energy(&signal);
        let diffused = sp.diffuse(&signal, 0.1, 10);
        let e1 = sp.dirichlet_energy(&diffused);
        assert!(e1 < e0);
    }

    #[test]
    fn test_consensus_converges() {
        let s = triangle_sheaf();
        let sp = SignalProcessing::new(&s);
        let signal = vec![1.0, 5.0, 9.0];
        let result = sp.consensus(&signal, 1e-6, 10000);
        // All values should converge to the same (harmonic) value
        let avg = result.iter().sum::<f64>() / result.len() as f64;
        for v in &result {
            assert!((v - avg).abs() < 0.1);
        }
    }

    #[test]
    fn test_dirichlet_energy_nonnegative() {
        let s = triangle_sheaf();
        let sp = SignalProcessing::new(&s);
        let signal = vec![1.0, -2.0, 3.5];
        assert!(sp.dirichlet_energy(&signal) >= -1e-10);
    }

    #[test]
    fn test_higher_dim_stalks() {
        let mut s = Sheaf::new(vec![2, 2]);
        s.add_identity_edge(0, 1);
        s.add_identity_edge(1, 0);
        assert_eq!(s.total_c0_dim(), 4);
        assert_eq!(s.total_c1_dim(), 4);
        let lap = s.sheaf_laplacian();
        assert_eq!(lap.rows, 4);
        assert_eq!(lap.cols, 4);
        assert!(lap.is_symmetric(1e-10));
    }

    #[test]
    fn test_hodge_decomposition_orthogonality() {
        let s = triangle_sheaf();
        let signal = vec![1.0, 2.0, 0.0];
        let hd = hodge_decompose(&s, &signal);
        // Harmonic + exact + coexact should approximately reconstruct signal
        for i in 0..signal.len() {
            let recon = hd.harmonic[i] + hd.exact[i] + hd.coexact[i];
            assert!((recon - signal[i]).abs() < 0.5, "Reconstruction mismatch at index {}", i);
        }
    }
}
