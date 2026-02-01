# Alexandrov Puzzle: Latin Cross Folding
# Enumerates all valid chuck folding sequences for a Latin cross
# and visualizes the 2D nets with glued vertices

import numpy as np
import copy
import matplotlib.pyplot as plt
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

class ChuckFolder:
    """Enumerates all valid chuck folding sequences."""

    def __init__(self, num_vertices, initial_angles):
        self.M = num_vertices
        self.initial_angles = initial_angles

    def run(self):
        """Run the enumeration and return (fold_sequences, equivalences)."""
        results_T = []
        results_Equ = []

        for start in range(self.M // 2):
            state = self._initial_state()
            if self._can_fold(state, start, 0):
                self._fold_recursive(state, start, 0, [], results_T, results_Equ)

        # Sort equivalence classes
        for equiv in results_Equ:
            for eq in equiv:
                eq.sort()
            equiv.sort()

        return results_T, results_Equ

    def _initial_state(self):
        """Create initial state for folding."""
        return {
            'Arg': [self.initial_angles.copy()] + [[] for _ in range(8)],
            'N': [list(range(1, self.M + 1))] + [[] for _ in range(8)],
            'V': [[[i] for i in range(1, self.M + 1)]] + [[] for _ in range(7)],
        }

    def _can_fold(self, state, i, n=0):
        """Check if folding at vertex i is valid at depth n."""
        num_vertices = self.M - 2 * n
        i2 = (i - 1) % num_vertices
        i3 = (i + 1) % num_vertices
        if state['Arg'][n][i3] + state['Arg'][n][i2] < 361:
            return True
        elif n == self.M // 2 - 1:
            return True
        return False

    def _fold(self, state, i, n):
        """Perform one fold operation at vertex i, depth n."""
        m = self.M - 2 * n
        i1, i2, i3 = i % m, (i - 1) % m, (i + 1) % m

        arg = copy.deepcopy(state['Arg'][n])
        num = copy.deepcopy(state['N'][n])
        ver = copy.deepcopy(state['V'][n])

        # Update angles (glue i2 and i3)
        arg[i2] = arg[i3] = arg[i2] + arg[i3]

        # Merge equivalence classes for glued vertices
        merged = set(ver[num[i2] - 1]) | set(ver[num[i3] - 1])
        for v in merged:
            ver[v - 1] = list(merged)

        # Remove i1 and one adjacent (descending order to preserve indices)
        for idx in sorted([i1, i3 if i2 < i3 else i2], reverse=True):
            arg.pop(idx)
            num.pop(idx)

        arg.extend([0, 0])
        num.extend([0, 0])
        state['Arg'][n + 1] = arg
        state['N'][n + 1] = num
        state['V'][n + 1] = ver

    def _fold_recursive(self, state, i, n, fold_seq, results_T, results_Equ):
        """Recursively explore folding sequences."""
        self._fold(state, i, n)
        new_seq = fold_seq + [state['N'][n][i]]
        next_n = n + 1

        for j in range(self.M - 2 * next_n):
            state['Arg'][next_n + 2] = state['Arg'][next_n + 1]
            state['N'][next_n + 2] = state['N'][next_n + 1]
            if self._can_fold(state, j, next_n):
                if next_n == self.M // 2 - 1:
                    self._fold_final(state, j, next_n, new_seq, results_T, results_Equ)
                else:
                    self._fold_recursive(state, j, next_n, new_seq, results_T, results_Equ)

    def _fold_final(self, state, i, n, fold_seq, results_T, results_Equ):
        """Handle the final fold step."""
        self._fold(state, i, n)
        final_seq = fold_seq + [state['N'][n][i]]
        results_T.append(final_seq)
        results_Equ.append(copy.deepcopy(state['V'][n + 1]))


def run_enumeration():
    """Run the chuck folding enumeration."""
    folder = ChuckFolder(M, INITIAL_ANGLES)
    return folder.run()


def classify_results(T, Equ):
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
    Classify polyhedron type based on the multiset of angles at essential vertices.
    Essential vertices have angle sum < 360 degrees.
    Returns: 'T' (tetrahedron), 'P' (pentahedron), 'O' (octahedron), 'C' (cube), 'Q' (doubly covered pentagon)
    """
    # Calculate angle sums for each equivalence class
    angle_sums = []
    seen = set()
    for eq in equivalence:
        eq_tuple = tuple(sorted(eq))
        if eq_tuple not in seen:
            seen.add(eq_tuple)
            angle_sum = sum(INITIAL_ANGLES[v - 1] for v in eq)
            angle_sums.append(angle_sum)

    # Essential vertices: angle sum < 360 degrees (have positive angular deficit)
    essential_angles = tuple(sorted([a for a in angle_sums if a < 360]))

    # Classify based on the multiset of angles
    angle_to_type = {
        (90, 90, 270, 270): 'T',                          # Tetrahedron
        (90, 180, 180, 270): 'Q',                          # Doubly covered pentagon
        (90, 180, 270, 270, 270): 'P',                     # Pentahedron
        (180, 180, 270, 270, 270, 270): 'O',               # Octahedron
        (270, 270, 270, 270, 270, 270, 270, 270): 'C',     # Cube
    }

    return angle_to_type.get(essential_angles, '?')


# ============================================
# SVG Generation
# ============================================

def get_fold_lines_for_type(poly_type, variant=0):
    """Get fold lines for each polyhedron type and variant.
    Returns list of line segments: [((x1, y1), (x2, y2)), ...]
    Coordinates are mirrored (x -> 3-x) to match the algorithm's vertex numbering.
    """
    fold_lines = {
        # Q0 (26541ab): mirrored from TikZ
        ('Q', 0): [
            ((1, 3), (2, 3)),
            ((2, 3), (3, 2)),
            ((0, 3), (2, 1)),
            ((2, 1), (1, 0)),
        ],
        # Q1 (1235478): mirrored from TikZ
        ('Q', 1): [
            ((0, 2), (2, 4)),
            ((2, 2), (2, 3)),
            ((2, 2), (1, 1)),
            ((1, 1), (2, 0)),
        ],
        # T0 (1235487): mirrored from TikZ
        ('T', 0): [
            ((2, 4), (1, 2)),
            ((1, 2), (2, 2)),
            ((2, 2), (2, 3)),
            ((2, 2), (1, 0)),
            ((1, 0), (2, 2/3)),
            ((3, 2.5), (8/3, 3)),
            ((2, 4), (1, 10/3)),
            ((0.5, 3), (0, 8/3)),
        ],
        # T1 (25741ab): mirrored from TikZ
        ('T', 1): [
            ((1.5, 4), (1, 11/3)),
            ((0, 3), (2, 2)),
            ((2, 2), (2, 3)),
            ((2, 3), (1, 3)),
            ((2, 3), (3, 2.5)),
            ((0, 3), (2/3, 2)),
            ((1, 1.5), (2, 0)),
            ((2, 0), (1, 0.5)),
        ],
        # P0 (1236549): mirrored from TikZ
        ('P', 0): [
            ((1, 3.5), (2, 4)),
            ((2, 4), (1, 2)),
            ((1, 2), (2, 1)),
            ((2, 1), (1, 0)),
            ((1, 1), (2, 1)),
            ((2, 2), (2, 3)),
            ((3, 2), (2.5, 3)),
        ],
        # P1 (3216789): mirrored from TikZ
        ('P', 1): [
            ((0, 3), (1, 2)),
            ((1, 2), (3, 3)),
            ((3, 3), (2.5, 2)),
            ((1, 3), (2, 3)),
            ((1, 4), (2, 3.5)),
            ((1, 1), (2, 1)),
            ((2, 1), (1, 0)),
        ],
        # O0 (2154789): mirrored from TikZ
        ('O', 0): [
            ((2, 4), (1, 3.5)),
            ((1, 0), (2, 0.5)),
            ((2, 1), (1, 1)),
            ((1, 1), (3, 3)),
            ((1, 3), (2, 3)),
            ((2, 3), (2, 2)),
            ((2, 2), (1, 2)),
            ((1, 2), (2, 3)),
            ((1, 2), (0, 3)),
        ],
        # O1 (2165a94): mirrored from TikZ
        ('O', 1): [
            ((2, 4), (1, 3)),
            ((1, 3), (2, 3)),
            ((2, 3), (1, 2)),
            ((1, 2), (1, 3)),
            ((1, 3), (0, 2)),
            ((2, 3), (2, 2)),
            ((3, 3), (2.5, 2)),
            ((1, 2), (2, 1)),
            ((2, 1), (1, 1)),
            ((2, 1), (1.5, 0)),
        ],
    }

    return fold_lines.get((poly_type, variant), [])


def generate_svg(poly_type, variant=0, scale=50):
    """Generate SVG string for a Latin cross net."""
    fold_lines = get_fold_lines_for_type(poly_type, variant)

    # SVG dimensions
    width = 4 * scale
    height = 5 * scale
    margin = 10

    # Latin cross outline path
    outline_points = [
        (2, 4), (2, 3), (3, 3), (3, 2), (2, 2), (2, 1), (2, 0),
        (1, 0), (1, 1), (1, 2), (0, 2), (0, 3), (1, 3), (1, 4)
    ]

    # Internal grid lines (always present)
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
        fold_lines_svg += f'  <line x1="{sx1}" y1="{sy1}" x2="{sx2}" y2="{sy2}" stroke="red" stroke-width="1.5"/>\n'

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width + 2*margin}" height="{height + 2*margin}" viewBox="0 0 {width + 2*margin} {height + 2*margin}">
  <!-- Outline (black) -->
  <path d="{outline_path}" fill="none" stroke="black" stroke-width="2"/>
  <!-- Grid lines (black) -->
{grid_lines_svg}
  <!-- Fold lines (red) -->
{fold_lines_svg}
  <!-- Label -->
  <text x="{width//2 + margin}" y="{height + margin + 5}" text-anchor="middle" font-family="serif" font-size="14" font-style="italic">{poly_type}<tspan baseline-shift="sub" font-size="10">{variant}</tspan></text>
</svg>'''
    return svg


def save_svg(poly_type, variant, filename):
    """Save SVG to file."""
    svg_content = generate_svg(poly_type, variant)
    with open(filename, 'w') as f:
        f.write(svg_content)
    print(f"Saved: {filename}")


def generate_net_svg(equivalence, title="", scale=50, variant=0):
    """Generate SVG string for a specific folding pattern."""
    poly_type = classify_polyhedron(equivalence)
    fold_lines = get_fold_lines_for_type(poly_type, variant)

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


def save_net_svg(equivalence, filename, title="", variant=0):
    """Save net SVG to file."""
    svg_content = generate_net_svg(equivalence, title, variant=variant)
    with open(filename, 'w') as f:
        f.write(svg_content)
    print(f"Saved: {filename}")


# ============================================
# Main
# ============================================

def main():
    # Run enumeration
    T, Equ = run_enumeration()

    # Classify and remove duplicates
    results = classify_results(T, Equ)

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
        print(f"     Total deficit: {sum(360 - a for v, a in essential)}°\n")

    # Generate visualizations
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate SVG files for each pattern
    fold_to_variant = {
        (2, 6, 5, 4, 1, 10, 11): ('Q', 0),  # Q0: 26541ab
        (1, 2, 3, 5, 4, 7, 8): ('Q', 1),    # Q1: 1235478
        (1, 2, 3, 5, 4, 8, 7): ('T', 0),    # T0: 1235487
        (2, 5, 7, 4, 1, 10, 11): ('T', 1),  # T1: 25741ab
        (1, 2, 3, 6, 5, 4, 9): ('P', 0),    # P0: 1236549
        (3, 2, 1, 6, 7, 8, 9): ('P', 1),    # P1: 3216789
        (2, 1, 5, 4, 7, 8, 9): ('O', 0),    # O0: 2154789
        (2, 1, 6, 5, 10, 9, 4): ('O', 1),   # O1: 2165a94
    }

    print("\nGenerating SVG files...")
    type_counts = {}
    for i, (fold_seq, equiv) in enumerate(results):
        poly_type = classify_polyhedron(equiv)
        fold_key = tuple(fold_seq)
        if fold_key in fold_to_variant:
            _, variant = fold_to_variant[fold_key]
        else:
            variant = type_counts.get(poly_type, 0)
            type_counts[poly_type] = variant + 1
        filename = os.path.join(output_dir, f"{poly_type}{variant}.svg")
        title = f"{poly_type}{variant}: {fold_seq}"
        save_net_svg(equiv, filename, title, variant=variant)

    print(f"\nOutput saved to: {output_dir}/")


if __name__ == "__main__":
    main()
