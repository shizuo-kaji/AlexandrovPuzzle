# Alexandrov Puzzle: Latin Cross Folding
# Enumerates all valid chuck folding sequences for a Latin cross
# and visualizes the 2D nets with glued vertices

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import os

# ============================================
# Latin Cross Configuration
# ============================================

M = 14  # Number of vertices

# Vertex coordinates (1-indexed)
VERTEX_COORDS = {
    1:  (2, 4), 2:  (2, 3), 3:  (3, 3), 4:  (3, 2),
    5:  (2, 2), 6:  (2, 1), 7:  (2, 0), 8:  (1, 0),
    9:  (1, 1), 10: (1, 2), 11: (0, 2), 12: (0, 3),
    13: (1, 3), 14: (1, 4)
}

# Outline vertex order (closed loop)
OUTLINE_VERTICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1]

# Initial angles at each vertex (degrees)
INITIAL_ANGLES = [90, 270, 90, 90, 270, 180, 90, 90, 180, 270, 90, 90, 270, 90]


# ============================================
# Chuck Folding Algorithm
# ============================================

class UnionFind:
    """Union-Find data structure with path compression for equivalence tracking."""

    def __init__(self, n):
        self.parent = np.arange(n + 1, dtype=np.int32)  # 1-indexed

    def find(self, x):
        """Find root with path compression."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, x, y):
        """Unite two sets."""
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

    def get_equivalence_classes(self):
        """Convert to list of equivalence classes for output compatibility."""
        classes = {}
        for i in range(1, len(self.parent)):
            root = self.find(i)
            if root not in classes:
                classes[root] = []
            classes[root].append(i)
        return [classes[self.find(i)] for i in range(1, len(self.parent))]

    def copy(self):
        """Create a copy of this UnionFind."""
        new_uf = UnionFind.__new__(UnionFind)
        new_uf.parent = self.parent.copy()
        return new_uf


class ChuckFolder:
    """Enumerates all valid chuck folding sequences using numpy arrays."""

    def __init__(self, num_vertices, initial_angles):
        self.M = num_vertices
        self.initial_angles = np.array(initial_angles, dtype=np.int32)
        self.max_depth = num_vertices // 2 + 2

    def run(self):
        """Run the enumeration and return (fold_sequences, equivalences)."""
        results_T = []
        results_Equ = []

        for start in range(self.M // 2):
            arg, num, uf = self._initial_state()
            if self._can_fold(arg, 0, start):
                self._fold_recursive(arg, num, uf, start, 0, [], results_T, results_Equ)

        # Sort equivalence classes
        for equiv in results_Equ:
            for eq in equiv:
                eq.sort()
            equiv.sort()

        return results_T, results_Equ

    def _initial_state(self):
        """Create initial state using numpy arrays."""
        arg = np.zeros((self.max_depth, self.M), dtype=np.int32)
        num = np.zeros((self.max_depth, self.M), dtype=np.int32)
        arg[0, :] = self.initial_angles
        num[0, :] = np.arange(1, self.M + 1)
        uf = UnionFind(self.M)
        return arg, num, uf

    def _can_fold(self, arg, n, i):
        """Check if folding at vertex i is valid at depth n."""
        num_vertices = self.M - 2 * n
        i2 = (i - 1) % num_vertices
        i3 = (i + 1) % num_vertices
        if arg[n, i3] + arg[n, i2] < 361:
            return True
        elif n == self.M // 2 - 1:
            return True
        return False

    def _fold(self, arg, num, uf, i, n):
        """Perform one fold operation at vertex i, depth n."""
        m = self.M - 2 * n
        i1, i2, i3 = i % m, (i - 1) % m, (i + 1) % m

        # Copy current level to next level
        new_arg = arg[n, :m].copy()
        new_num = num[n, :m].copy()

        # Update angles (glue i2 and i3)
        combined_angle = new_arg[i2] + new_arg[i3]
        new_arg[i2] = combined_angle
        new_arg[i3] = combined_angle

        # Merge equivalence classes
        uf.union(new_num[i2], new_num[i3])

        # Remove i1 and one adjacent using numpy delete
        idx_to_remove = sorted([i1, i3 if i2 < i3 else i2], reverse=True)
        new_arg = np.delete(new_arg, idx_to_remove)
        new_num = np.delete(new_num, idx_to_remove)

        # Store in next level
        new_m = m - 2
        arg[n + 1, :new_m] = new_arg
        num[n + 1, :new_m] = new_num

    def _fold_recursive(self, arg, num, uf, i, n, fold_seq, results_T, results_Equ):
        """Recursively explore folding sequences."""
        # Save state for backtracking
        uf_backup = uf.copy()
        arg_backup = arg[n + 1].copy()
        num_backup = num[n + 1].copy()

        self._fold(arg, num, uf, i, n)
        new_seq = fold_seq + [int(num[n, i])]
        next_n = n + 1

        for j in range(self.M - 2 * next_n):
            if self._can_fold(arg, next_n, j):
                if next_n == self.M // 2 - 1:
                    self._fold_final(arg, num, uf, j, next_n, new_seq, results_T, results_Equ)
                else:
                    self._fold_recursive(arg, num, uf, j, next_n, new_seq, results_T, results_Equ)

        # Restore state for backtracking
        uf.parent[:] = uf_backup.parent
        arg[n + 1] = arg_backup
        num[n + 1] = num_backup

    def _fold_final(self, arg, num, uf, i, n, fold_seq, results_T, results_Equ):
        """Handle the final fold step."""
        uf_backup = uf.copy()

        self._fold(arg, num, uf, i, n)
        final_seq = fold_seq + [int(num[n, i])]
        results_T.append(final_seq)
        results_Equ.append(uf.get_equivalence_classes())

        # Restore state
        uf.parent[:] = uf_backup.parent

def remove_mirrors(T, Equ):
    """Remove symmetric duplicates from results."""
    # Compute mirror images (left-right symmetry: vertex i -> 15-i)
    mirrors = []
    for equiv in Equ:
        mirrored = []
        for eq in equiv:
            mirrored.append(sorted([15 - v for v in eq]))
        mirrors.append(sorted(mirrored))

    # Find unique patterns
    unique_indices = [0]
    for i in range(1, len(T)):
        is_duplicate = False
        for j in unique_indices:
            if Equ[j] == Equ[i] or Equ[j] == mirrors[i]:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)

    return [(T[i], Equ[i]) for i in unique_indices]


# ============================================
# Polyhedron Classification
# ============================================

def classify_polyhedron(equivalence):
    """
    Classify polyhedron type based on number of essential vertices and deficit pattern.

    Returns:
        'C' - Cube (8 vertices)
        'T' - Tetrahedron (4 vertices)
        'Q' - Double-covered quadrilateral (4 vertices)
        'P' - Pentahedron (5 vertices)
        'O' - Octahedron (6 vertices)
    """
    essential_vertices = get_essential_vertices(equivalence)
    n = len(essential_vertices)

    if n < 4:
        return 'C'  # Degenerate case

    deficits = sorted([d for _, _, d in essential_vertices])

    if n == 4:
        # Distinguish T (tetrahedron) from Q (doubly-covered quadrilateral)
        # T: symmetric deficits like [90, 90, 270, 270] - 2 distinct values
        # Q: asymmetric deficits like [90, 180, 180, 270] - 3 distinct values
        unique_deficits = len(set(deficits))
        if unique_deficits == 2:
            return 'T'  # Tetrahedron (symmetric)
        else:
            return 'Q'  # Doubly-covered quadrilateral (asymmetric)
    elif n == 5:
        return 'P'  # Pentahedron
    elif n == 6:
        return 'O'  # Octahedron
    elif n == 8:
        return 'C'  # Cube
    else:
        return 'C'  # Default for unexpected cases


# ============================================
# 3D Polyhedron Reconstruction
# ============================================

def get_essential_vertices(equivalence):
    """
    Get essential vertices (those with angular deficit > 0).
    Returns list of (equivalence_class, angle_sum, deficit) tuples.
    """
    essential = []
    seen = set()
    for eq in equivalence:
        eq_tuple = tuple(sorted(eq))
        if eq_tuple not in seen:
            seen.add(eq_tuple)
            angle_sum = sum(INITIAL_ANGLES[v - 1] for v in eq)
            if angle_sum < 360:
                deficit = 360 - angle_sum
                essential.append((list(eq_tuple), angle_sum, deficit))
    return essential


def get_net_edges():
    """
    Get all edges of the Latin cross net (boundary + internal grid lines).
    Returns list of (v1, v2, length) tuples.
    """
    edges = []

    # Boundary edges from outline
    outline = OUTLINE_VERTICES
    for i in range(len(outline) - 1):
        v1, v2 = outline[i], outline[i + 1]
        p1 = np.array(VERTEX_COORDS[v1])
        p2 = np.array(VERTEX_COORDS[v2])
        length = np.linalg.norm(p2 - p1)
        edges.append((v1, v2, length))

    # Internal grid edges (inside the cross)
    # These connect vertices across the interior
    internal_edges = [
        (9, 6),    # (1,1) to (2,1)
        (10, 5),   # (1,2) to (2,2)
        (13, 2),   # (1,3) to (2,3)
        (10, 13),  # (1,2) to (1,3)
        (5, 2),    # (2,2) to (2,3)
    ]
    for v1, v2 in internal_edges:
        p1 = np.array(VERTEX_COORDS[v1])
        p2 = np.array(VERTEX_COORDS[v2])
        length = np.linalg.norm(p2 - p1)
        edges.append((v1, v2, length))

    return edges


def point_in_cross(x, y):
    """Check if point (x, y) is inside or on the boundary of the Latin cross."""
    eps = 1e-9
    # Vertical bar: 1 <= x <= 2, 0 <= y <= 4
    in_vertical = (1 - eps <= x <= 2 + eps) and (0 - eps <= y <= 4 + eps)
    # Horizontal bar: 0 <= x <= 3, 2 <= y <= 3
    in_horizontal = (0 - eps <= x <= 3 + eps) and (2 - eps <= y <= 3 + eps)
    return in_vertical or in_horizontal


def segment_in_cross(p1, p2, n_samples=50):
    """Check if line segment from p1 to p2 is entirely within the Latin cross."""
    x1, y1 = p1
    x2, y2 = p2
    for i in range(n_samples + 1):
        t = i / n_samples
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if not point_in_cross(x, y):
            return False
    return True


def get_glued_edges(equivalence):
    """
    Find pairs of boundary edges that are glued together.
    Two edges (a,b) and (c,d) are glued if a↔c and b↔d (or a↔d and b↔c).
    Returns list of ((a,b), (c,d), orientation) where orientation is 1 if a↔c,b↔d or -1 if a↔d,b↔c.
    """
    # Build vertex-to-class mapping
    vertex_class = {}
    for eq in equivalence:
        eq_frozen = frozenset(eq)
        for v in eq:
            vertex_class[v] = eq_frozen

    # Get boundary edges
    boundary_edges = []
    outline = OUTLINE_VERTICES
    for i in range(len(outline) - 1):
        boundary_edges.append((outline[i], outline[i + 1]))

    # Find glued edge pairs
    glued_pairs = []
    for i, (a, b) in enumerate(boundary_edges):
        for j, (c, d) in enumerate(boundary_edges):
            if j <= i:
                continue
            # Check edge lengths match
            pa, pb = np.array(VERTEX_COORDS[a]), np.array(VERTEX_COORDS[b])
            pc, pd = np.array(VERTEX_COORDS[c]), np.array(VERTEX_COORDS[d])
            len_ab = np.linalg.norm(pb - pa)
            len_cd = np.linalg.norm(pd - pc)
            if abs(len_ab - len_cd) > 0.01:
                continue

            # Check if a↔c and b↔d (same orientation)
            if vertex_class.get(a) == vertex_class.get(c) and vertex_class.get(b) == vertex_class.get(d):
                glued_pairs.append(((a, b), (c, d), 1))
            # Check if a↔d and b↔c (opposite orientation)
            elif vertex_class.get(a) == vertex_class.get(d) and vertex_class.get(b) == vertex_class.get(c):
                glued_pairs.append(((a, b), (c, d), -1))

    return glued_pairs


def geodesic_through_glued_edge(p1, p2, edge1, edge2, orientation):
    """
    Compute geodesic from p1 to p2 crossing through glued edges.
    edge1 = (a, b), edge2 = (c, d) with vertices glued according to orientation.
    Returns (distance, crossing_point1, crossing_point2) or None if path is longer.
    """
    a, b = np.array(VERTEX_COORDS[edge1[0]], dtype=float), np.array(VERTEX_COORDS[edge1[1]], dtype=float)
    c, d = np.array(VERTEX_COORDS[edge2[0]], dtype=float), np.array(VERTEX_COORDS[edge2[1]], dtype=float)
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)

    if orientation == -1:
        c, d = d, c  # Swap to match orientation

    # Point at parameter t on edge1: a + t*(b-a)
    # Corresponding point on edge2: c + t*(d-c)
    # Total distance: |p1 - (a + t*(b-a))| + |p2 - (c + t*(d-c))|

    # Use golden section search to minimize
    def path_length(t):
        pt1 = a + t * (b - a)
        pt2 = c + t * (d - c)
        return np.linalg.norm(p1 - pt1) + np.linalg.norm(p2 - pt2)

    # Golden section search
    phi = (1 + np.sqrt(5)) / 2
    tol = 1e-6
    lo, hi = 0.0, 1.0
    c1 = hi - (hi - lo) / phi
    c2 = lo + (hi - lo) / phi
    f1, f2 = path_length(c1), path_length(c2)

    while hi - lo > tol:
        if f1 < f2:
            hi = c2
            c2 = c1
            f2 = f1
            c1 = hi - (hi - lo) / phi
            f1 = path_length(c1)
        else:
            lo = c1
            c1 = c2
            f1 = f2
            c2 = lo + (hi - lo) / phi
            f2 = path_length(c2)

    t_opt = (lo + hi) / 2
    crossing1 = tuple(a + t_opt * (b - a))
    crossing2 = tuple(c + t_opt * (d - c))
    dist = path_length(t_opt)

    # Check if both segments are in the cross
    if segment_in_cross(tuple(p1), crossing1) and segment_in_cross(crossing2, tuple(p2)):
        return (dist, crossing1, crossing2)
    return None


def build_geodesic_graph(equivalence):
    """
    Build adjacency matrix for geodesic distances on the folded surface.
    Considers:
    - Direct lines within the net
    - Paths through glued vertices
    - Geodesics crossing glued edges at non-vertex points
    """
    n = M  # 14 vertices

    # Initialize distance matrix with infinity
    dist = np.full((n + 1, n + 1), np.inf)  # 1-indexed
    np.fill_diagonal(dist, 0)

    # Add all net edges (boundary + internal)
    for v1, v2, length in get_net_edges():
        dist[v1, v2] = min(dist[v1, v2], length)
        dist[v2, v1] = min(dist[v2, v1], length)

    # Add direct lines through face interiors (geodesics on flat surface)
    for v1 in range(1, n + 1):
        for v2 in range(v1 + 1, n + 1):
            p1 = VERTEX_COORDS[v1]
            p2 = VERTEX_COORDS[v2]
            if segment_in_cross(p1, p2):
                eucl_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                dist[v1, v2] = min(dist[v1, v2], eucl_dist)
                dist[v2, v1] = min(dist[v2, v1], eucl_dist)

    # Add zero-distance edges for glued vertices (same equivalence class)
    for eq in equivalence:
        for i, v1 in enumerate(eq):
            for v2 in eq[i + 1:]:
                dist[v1, v2] = 0
                dist[v2, v1] = 0

    # Consider geodesics through glued edges (crossing at non-vertex points)
    glued_edges = get_glued_edges(equivalence)
    for v1 in range(1, n + 1):
        for v2 in range(v1 + 1, n + 1):
            p1 = VERTEX_COORDS[v1]
            p2 = VERTEX_COORDS[v2]
            for edge1, edge2, orientation in glued_edges:
                result = geodesic_through_glued_edge(p1, p2, edge1, edge2, orientation)
                if result:
                    d, _, _ = result
                    if d < dist[v1, v2]:
                        dist[v1, v2] = d
                        dist[v2, v1] = d
                # Also try the reverse direction (p2 to p1 through the gluing)
                result = geodesic_through_glued_edge(p2, p1, edge1, edge2, orientation)
                if result:
                    d, _, _ = result
                    if d < dist[v1, v2]:
                        dist[v1, v2] = d
                        dist[v2, v1] = d

    # Floyd-Warshall for all-pairs shortest paths
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist


def reconstruct_3d_polyhedron(equivalence):
    """
    Reconstruct 3D polyhedron coordinates from the folding pattern.

    Returns:
        vertices_3d: numpy array of shape (n_vertices, 3)
        edges: list of (i, j) tuples (from convex hull)
        vertex_labels: list of equivalence classes for each vertex
    """
    essential = get_essential_vertices(equivalence)
    n_vertices = len(essential)

    if n_vertices < 4:
        return None, None, None

    # Build geodesic distance matrix on the folded surface
    full_dist = build_geodesic_graph(equivalence)

    # Extract distances between essential vertices
    representatives = [eq[0] for eq, _, _ in essential]
    dist_matrix = np.zeros((n_vertices, n_vertices))
    for i in range(n_vertices):
        for j in range(n_vertices):
            dist_matrix[i, j] = full_dist[representatives[i], representatives[j]]

    # Use classical MDS (multidimensional scaling) for initial embedding
    # Classical MDS using eigendecomposition of the Gram matrix
    def classical_mds(D, n_dims=3):
        """Classical MDS: find coordinates from distance matrix."""
        n = D.shape[0]
        # Double centering
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Sort by eigenvalue descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Take top n_dims dimensions
        # Handle negative eigenvalues (non-Euclidean distances)
        eigvals_pos = np.maximum(eigvals[:n_dims], 0)
        coords = eigvecs[:, :n_dims] * np.sqrt(eigvals_pos)
        return coords

    vertices_3d = classical_mds(dist_matrix, 3)

    # Center at origin
    vertices_3d -= vertices_3d.mean(axis=0)

    # Compute edges from convex hull
    edges = []
    try:
        if n_vertices >= 4:
            hull = ConvexHull(vertices_3d)
            # Extract edges from hull simplices (triangular faces)
            edge_set = set()
            for simplex in hull.simplices:
                for k in range(3):
                    i, j = simplex[k], simplex[(k + 1) % 3]
                    edge_set.add((min(i, j), max(i, j)))
            edges = list(edge_set)
    except Exception:
        # Fall back to complete graph edges
        edges = [(i, j) for i in range(n_vertices) for j in range(i + 1, n_vertices)]

    vertex_labels = [eq for eq, _, _ in essential]

    return vertices_3d, edges, vertex_labels


def plot_3d_polyhedron(vertices_3d, edges, vertex_labels, title="", ax=None):
    """Plot the 3D polyhedron using matplotlib."""
    if vertices_3d is None:
        return None

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Plot vertices
    ax.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2],
               s=100, c='blue', alpha=0.8)

    # Label vertices
    for i, label in enumerate(vertex_labels):
        ax.text(vertices_3d[i, 0], vertices_3d[i, 1], vertices_3d[i, 2],
                f'  {label}', fontsize=8)

    # Plot edges
    for i, j in edges:
        ax.plot([vertices_3d[i, 0], vertices_3d[j, 0]],
                [vertices_3d[i, 1], vertices_3d[j, 1]],
                [vertices_3d[i, 2], vertices_3d[j, 2]],
                'k-', linewidth=1.5)

    # Try to draw faces using convex hull
    try:
        if len(vertices_3d) >= 4:
            hull = ConvexHull(vertices_3d)
            for simplex in hull.simplices:
                triangle = vertices_3d[simplex]
                ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                                alpha=0.3, color='cyan')
    except Exception:
        pass

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.ptp(vertices_3d, axis=0).max() / 2
    mid = vertices_3d.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    return fig


def save_polyhedron_stl(vertices_3d, filename, solid_name="polyhedron"):
    """Save polyhedron to STL file format (ASCII)."""
    if vertices_3d is None or len(vertices_3d) < 4:
        return

    try:
        hull = ConvexHull(vertices_3d)
    except Exception:
        return

    with open(filename, 'w') as f:
        f.write(f"solid {solid_name}\n")

        for simplex in hull.simplices:
            # Get the three vertices of the triangle
            v0 = vertices_3d[simplex[0]]
            v1 = vertices_3d[simplex[1]]
            v2 = vertices_3d[simplex[2]]

            # Compute normal vector using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_length = np.linalg.norm(normal)
            if norm_length > 1e-10:
                normal = normal / norm_length
            else:
                normal = np.array([0.0, 0.0, 1.0])

            # Write facet
            f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write(f"    outer loop\n")
            f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write(f"    endloop\n")
            f.write(f"  endfacet\n")

        f.write(f"endsolid {solid_name}\n")

    print(f"Saved: {filename}")


# ============================================
# SVG Generation
# ============================================

def compute_fold_lines(equivalence):
    """
    Compute fold lines from the 3D polyhedron structure.
    Each 3D edge corresponds to a geodesic on the folded surface.
    Geodesics appear as piecewise straight lines on the 2D net,
    going through glued vertices.

    Only includes edges with dihedral angle < 180° (real folded edges).

    Returns list of line segments: [((x1, y1), (x2, y2)), ...]
    """
    # Get 3D structure
    vertices_3d, edges_3d, vertex_labels = reconstruct_3d_polyhedron(equivalence)

    if vertices_3d is None or edges_3d is None:
        return []

    # Compute dihedral angles to filter out flat edges (angle = 180°)
    try:
        hull = ConvexHull(vertices_3d)

        # Build edge to faces mapping
        edge_faces = {}
        for face_idx, simplex in enumerate(hull.simplices):
            for k in range(3):
                i, j = simplex[k], simplex[(k + 1) % 3]
                edge = (min(i, j), max(i, j))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(face_idx)

        # Compute dihedral angle for each edge
        def compute_dihedral_angle(edge):
            """Compute dihedral angle between two faces sharing an edge."""
            faces = edge_faces.get(edge, [])
            if len(faces) != 2:
                return 180.0  # Boundary edge or degenerate

            # Get face normals (pointing outward from hull equations)
            n1 = hull.equations[faces[0]][:3]
            n2 = hull.equations[faces[1]][:3]

            # Dihedral angle: angle between outward normals
            # For convex hull, normals point outward, so dihedral = 180 - angle_between_normals
            cos_angle = np.clip(np.dot(n1, n2), -1, 1)
            angle_between = np.degrees(np.arccos(cos_angle))
            dihedral = 180.0 - angle_between

            return dihedral

        # Filter edges: keep only those with dihedral angle significantly < 180°
        real_edges = []
        for edge in edges_3d:
            edge_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            dihedral = compute_dihedral_angle(edge_key)
            if dihedral < 179.0:  # Not flat (with small tolerance)
                real_edges.append(edge)

        edges_3d = real_edges

    except Exception:
        pass  # Keep all edges if hull computation fails

    # Get geodesic distances
    full_dist = build_geodesic_graph(equivalence)

    # Build adjacency for path reconstruction (includes direct lines through faces)
    adj = {i: {} for i in range(1, M + 1)}
    # Add net edges
    for v1, v2, length in get_net_edges():
        adj[v1][v2] = length
        adj[v2][v1] = length
    # Add direct lines through face interiors
    for v1 in range(1, M + 1):
        for v2 in range(v1 + 1, M + 1):
            p1 = VERTEX_COORDS[v1]
            p2 = VERTEX_COORDS[v2]
            if segment_in_cross(p1, p2):
                eucl_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if v2 not in adj[v1] or eucl_dist < adj[v1][v2]:
                    adj[v1][v2] = eucl_dist
                    adj[v2][v1] = eucl_dist
    # Add zero-distance edges for glued vertices
    for eq in equivalence:
        for i, v1 in enumerate(eq):
            for v2 in eq[i + 1:]:
                adj[v1][v2] = 0
                adj[v2][v1] = 0

    def find_shortest_path(start, targets):
        """Find shortest path using Dijkstra with path reconstruction."""
        import heapq
        target_set = set(targets)
        dist = {i: float('inf') for i in range(1, M + 1)}
        prev = {i: None for i in range(1, M + 1)}
        dist[start] = 0
        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u in target_set:
                path = []
                curr = u
                while curr is not None:
                    path.append(curr)
                    curr = prev[curr]
                return path[::-1]
            for v, w in adj[u].items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
        return []

    fold_lines = []
    seen_edges = set()

    def add_line(p1, p2):
        """Add a fold line, avoiding duplicates."""
        if p1 == p2:
            return
        edge_key = (min(p1, p2), max(p1, p2))
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            fold_lines.append((p1, p2))

    # Get glued edges for geodesic computation
    glued_edges = get_glued_edges(equivalence)

    # For each 3D edge, find the geodesic path on the 2D net
    for idx_i, idx_j in edges_3d:
        eq_i = vertex_labels[idx_i]
        eq_j = vertex_labels[idx_j]
        target_dist = full_dist[eq_i[0], eq_j[0]]

        best_segments = None
        best_dist = float('inf')

        # Option 1: Try direct lines between representative vertices
        for vi in eq_i:
            for vj in eq_j:
                pi = VERTEX_COORDS[vi]
                pj = VERTEX_COORDS[vj]
                eucl_dist = np.sqrt((pi[0] - pj[0])**2 + (pi[1] - pj[1])**2)

                if abs(eucl_dist - target_dist) < 0.01 and segment_in_cross(pi, pj):
                    if eucl_dist < best_dist:
                        best_dist = eucl_dist
                        best_segments = [(pi, pj)]

        # Option 2: Try geodesics through glued edges (crossing at non-vertex points)
        for vi in eq_i:
            for vj in eq_j:
                pi = VERTEX_COORDS[vi]
                pj = VERTEX_COORDS[vj]
                for edge1, edge2, orientation in glued_edges:
                    result = geodesic_through_glued_edge(pi, pj, edge1, edge2, orientation)
                    if result:
                        d, cross1, cross2 = result
                        if abs(d - target_dist) < 0.01 and d < best_dist:
                            best_dist = d
                            best_segments = [(pi, cross1), (cross2, pj)]
                    # Also try reverse direction
                    result = geodesic_through_glued_edge(pj, pi, edge1, edge2, orientation)
                    if result:
                        d, cross1, cross2 = result
                        if abs(d - target_dist) < 0.01 and d < best_dist:
                            best_dist = d
                            best_segments = [(pj, cross1), (cross2, pi)]

        # Option 3: Fall back to vertex-path through gluings
        if best_segments is None:
            for vi in eq_i:
                path = find_shortest_path(vi, eq_j)
                if path:
                    path_dist = sum(
                        np.sqrt((VERTEX_COORDS[path[k]][0] - VERTEX_COORDS[path[k+1]][0])**2 +
                                (VERTEX_COORDS[path[k]][1] - VERTEX_COORDS[path[k+1]][1])**2)
                        for k in range(len(path) - 1)
                    )
                    if path_dist < best_dist:
                        best_dist = path_dist
                        segments = []
                        for k in range(len(path) - 1):
                            v1, v2 = path[k], path[k + 1]
                            if adj[v1][v2] >= 0.01:  # Skip glued vertex edges
                                segments.append((VERTEX_COORDS[v1], VERTEX_COORDS[v2]))
                        best_segments = segments

        # Add the best segments found
        if best_segments:
            for p1, p2 in best_segments:
                add_line(p1, p2)

    return fold_lines


def generate_net_svg(equivalence, title="", scale=50):
    """Generate SVG string for a specific folding pattern."""
    fold_lines = compute_fold_lines(equivalence)

    # SVG dimensions
    width = 4 * scale
    height = 5 * scale
    margin = 15

    # Latin cross outline points
    outline_points = [
        (2, 4), (2, 3), (3, 3), (3, 2), (2, 2), (2, 1), (2, 0),
        (1, 0), (1, 1), (1, 2), (0, 2), (0, 3), (1, 3), (1, 4)
    ]

    # Internal grid lines
    grid_lines = [
        ((1, 1), (2, 1)),
        ((1, 2), (2, 2)),
        ((1, 3), (2, 3)),
        ((1, 2), (1, 3)),
        ((2, 2), (2, 3)),
    ]

    def to_svg(x, y):
        return (margin + x * scale, margin + (4 - y) * scale)

    # Build outline path
    outline_path = "M " + " L ".join(f"{to_svg(x, y)[0]},{to_svg(x, y)[1]}" for x, y in outline_points) + " Z"

    # Build grid lines
    grid_lines_svg = ""
    for (x1, y1), (x2, y2) in grid_lines:
        sx1, sy1 = to_svg(x1, y1)
        sx2, sy2 = to_svg(x2, y2)
        grid_lines_svg += f'  <line x1="{sx1}" y1="{sy1}" x2="{sx2}" y2="{sy2}" stroke="black" stroke-width="1"/>\n'

    # Build fold lines
    fold_lines_svg = ""
    for (x1, y1), (x2, y2) in fold_lines:
        sx1, sy1 = to_svg(x1, y1)
        sx2, sy2 = to_svg(x2, y2)
        # Round to avoid floating point artifacts
        sx1, sy1 = round(sx1, 2), round(sy1, 2)
        sx2, sy2 = round(sx2, 2), round(sy2, 2)
        fold_lines_svg += f'  <line x1="{sx1}" y1="{sy1}" x2="{sx2}" y2="{sy2}" stroke="red" stroke-width="1.5"/>\n'

    # Color palette for equivalence classes
    colors = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # gray
    ]

    # Build vertex-to-color mapping based on equivalence classes
    vertex_color = {}
    seen = set()
    color_idx = 0
    for eq in equivalence:
        eq_tuple = tuple(sorted(eq))
        if eq_tuple not in seen:
            seen.add(eq_tuple)
            if len(eq) >= 2:
                # Glued vertices get a color
                color = colors[color_idx % len(colors)]
                color_idx += 1
                for v in eq:
                    vertex_color[v] = color
            else:
                # Single vertices are black
                vertex_color[eq[0]] = "black"

    # Build vertex dots with colors
    vertices_svg = ""
    for v, (x, y) in VERTEX_COORDS.items():
        sx, sy = to_svg(x, y)
        color = vertex_color.get(v, "black")
        vertices_svg += f'  <circle cx="{sx}" cy="{sy}" r="4" fill="{color}"/>\n'

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width + 2*margin}" height="{height + 2*margin + 20}" viewBox="0 0 {width + 2*margin} {height + 2*margin + 20}">
  <!-- Outline (black) -->
  <path d="{outline_path}" fill="none" stroke="black" stroke-width="2"/>
  <!-- Grid lines (black) -->
{grid_lines_svg}
  <!-- Fold lines (red) -->
{fold_lines_svg}
  <!-- Vertices (colored by equivalence class) -->
{vertices_svg}
  <!-- Title -->
  <text x="{width//2 + margin}" y="{height + 2*margin + 12}" text-anchor="middle" font-family="sans-serif" font-size="12">{title}</text>
</svg>'''
    return svg


def save_net_svg(equivalence, filename, title=""):
    """Save net SVG to file."""
    svg_content = generate_net_svg(equivalence, title)
    with open(filename, 'w') as f:
        f.write(svg_content)
    print(f"Saved: {filename}")


# ============================================
# Main
# ============================================

def main():
    # Run enumeration
    folder = ChuckFolder(M, INITIAL_ANGLES)
    T, Equ = folder.run()

    # Classify and remove duplicates
    results = remove_mirrors(T, Equ)

    # Print results
    print(f"Found {len(results)} unique folding patterns:\n")
    for i, (fold_seq, equiv) in enumerate(results):
        poly_type = classify_polyhedron(equiv)
        glued = sorted([tuple(eq) for eq in equiv if len(eq) >= 2])
        glued = [list(eq) for eq in set(glued)]
        glued.sort()

        # Calculate angle sum at each vertex
        angle_sums = []
        seen = set()
        for eq in equiv:
            eq_tuple = tuple(sorted(eq))
            if eq_tuple not in seen:
                seen.add(eq_tuple)
                # Sum angles for vertices in this equivalence class
                # INITIAL_ANGLES[i] corresponds to vertex i+1
                angle_sum = sum(INITIAL_ANGLES[v - 1] for v in eq)
                angle_sums.append((list(eq_tuple), angle_sum))

        # Essential vertices (angle sum < 360), sorted by angle in increasing order
        essential = [(v, a) for v, a in angle_sums if a < 360]
        essential.sort(key=lambda x: (x[1], x[0]))  # Sort by angle, then by vertex list

        print(f"  {i+1}. [{poly_type}] Fold: {fold_seq}")
        print(f"     Glued: {glued}")
        print(f"     Essential vertices: {essential}")

    # Generate visualizations
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations for each pattern
    print("\nGenerating visualizations...")
    type_counts = {}
    for i, (fold_seq, equiv) in enumerate(results):
        poly_type = classify_polyhedron(equiv)
        variant = type_counts.get(poly_type, 0)
        type_counts[poly_type] = variant + 1

        # Generate SVG
        filename = os.path.join(output_dir, f"{poly_type}{variant}.svg")
        title = f"{poly_type}{variant}: {fold_seq}"
        save_net_svg(equiv, filename, title)

        # Generate 3D polyhedron
        vertices_3d, edges, vertex_labels = reconstruct_3d_polyhedron(equiv)
        if vertices_3d is not None:
            # Save STL file
            stl_filename = os.path.join(output_dir, f"{poly_type}{variant}.stl")
            save_polyhedron_stl(vertices_3d, stl_filename, f"{poly_type}{variant}")

            # Save plot
            fig = plot_3d_polyhedron(vertices_3d, edges, vertex_labels, title)
            if fig:
                png_filename = os.path.join(output_dir, f"{poly_type}{variant}_3d.png")
                fig.savefig(png_filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved: {png_filename}")

    print(f"\nOutput saved to: {output_dir}/")


if __name__ == "__main__":
    main()
