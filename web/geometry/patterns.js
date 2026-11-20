// Corners are numbered clockwise from the top-right corner of the Latin cross.
// Other endpoints use coordinates measured in square side lengths, with y down.
// These are the creases in the Takenaga–Kaji folding diagrams.
export const chords = {
  C0: [
    [13, 2],
    [10, 5],
    [9, 6],
    [13, 10],
    [2, 5],
  ],
  T0: [
    [14, 5],
    [5, 10],
    [10, 13],
    [10, 7],
    [7, [1, 10 / 3]],
    [
      [0, 1.5],
      [1 / 3, 1],
    ],
    [14, [2, 2 / 3]],
    [
      [2.5, 1],
      [3, 4 / 3],
    ],
  ],
  T1: [
    [
      [1.5, 0],
      [2, 1 / 3],
    ],
    [3, 10],
    [10, 13],
    [13, 2],
    [13, [0, 1.5]],
    [3, [7 / 3, 2]],
    [[2, 2.5], 8],
    [8, [2, 3.5]],
  ],
  Q0: [
    [2, 13],
    [13, 11],
    [3, 9],
    [9, 7],
  ],
  Q1: [
    [4, 14],
    [10, 13],
    [10, 6],
    [6, 8],
  ],
  P0: [
    [[2, 0.5], 14],
    [14, 5],
    [5, 9],
    [9, 7],
    [6, 9],
    [10, 13],
    [11, [0.5, 1]],
  ],
  P1: [
    [3, 5],
    [5, 12],
    [12, [0.5, 2]],
    [2, 13],
    [1, [1, 0.5]],
    [6, 9],
    [9, 7],
  ],
  O0: [
    [14, [2, 0.5]],
    [7, [1, 3.5]],
    [9, 6],
    [6, 12],
    [2, 13],
    [13, 10],
    [10, 5],
    [5, 13],
    [5, 3],
  ],
  O1: [
    [14, 2],
    [2, 13],
    [13, 5],
    [5, 2],
    [2, 4],
    [13, 10],
    [12, [0.5, 2]],
    [5, 9],
    [9, 6],
    [9, [1.5, 4]],
  ],
};

// Exterior dihedral angles, measured from the flat sheet. Each letter in
// labels identifies a pair of unit perimeter edges that close in reverse
// directions. Positions are checked at both ends and at crease crossings.
const right = Math.PI / 2;
const tetraLong = Math.acos(-1 / 4),
  tetraTip = Math.acos(-5 / 8);
const pentaLong = Math.acos(-2 / 3),
  pentaShort = Math.acos(1 / 3);
const octaTip = Math.acos((5 * Math.SQRT2 - 7) / 8);
const octaA = Math.acos(2 - Math.SQRT2),
  octaB = Math.acos((Math.SQRT2 - 1) / 2);

export const folds = {
  C0: { labels: 'AABCCBDEFFEGGD', angles: Array(5).fill(right) },
  T0: {
    labels: 'ABCDDEECFFBAGG',
    angles: [tetraLong, Math.PI / 3, (2 * Math.PI) / 3, tetraLong, tetraTip, tetraTip, tetraTip, tetraTip],
  },
  T1: {
    labels: 'ABBACDEEFFDGGC',
    angles: [tetraTip, tetraLong, Math.PI / 3, (2 * Math.PI) / 3, tetraLong, tetraTip, tetraTip, tetraLong],
  },
  Q0: { labels: 'ABBACDEFFEDGGC', angles: Array(5).fill(Math.PI) },
  Q1: { labels: 'ABCDEEDCFFBAGG', angles: Array(5).fill(Math.PI) },
  P0: {
    labels: 'ABCDDCEFFEBAGG',
    angles: [pentaLong, pentaLong, pentaShort, pentaShort, right, right, pentaLong],
  },
  P1: {
    labels: 'ABCDDCBAEFGGFE',
    angles: [pentaShort, pentaLong, pentaLong, right, pentaLong, right, pentaShort],
  },
  O0: {
    labels: 'ABCDDCBAEEFGGF',
    angles: [octaTip, octaTip, octaA, octaA, octaB, octaA, octaA, octaB, octaA, octaB],
  },
  O1: {
    labels: 'AABCCBDEEDFGGF',
    angles: [octaB, octaA, octaA, octaB, octaA, octaA, octaTip, octaB, Math.PI - octaB, octaTip],
  },
};
