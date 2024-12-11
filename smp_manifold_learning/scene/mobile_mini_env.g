world: { X: "t (0 0 .2)" }

box (world): { shape:ssBox, Q: [2, 2, 1.5], size: [.3,.3,.1,.005], color: [1 .5 0], contact: 0 }

goal (world): { shape:ssBox, Q: [-1,-1, 1.3], size: [.3,.3,.1,.005], color: [.5], contact: 0 }

obs (world): { shape:ssBox, Q: [0.5, 0.5, 1.5], size: [.3,.3,.5,.005], color: [1 0 0], contact: 1 }