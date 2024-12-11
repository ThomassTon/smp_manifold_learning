base_origin: { X: [0, 0, .2] }

base (base_origin): {
 joint: transXYPhi, limits: [-10,10,-10,10,-4,4],
#, shape: ssCvx, mesh: <base.arr>, meshscale: .3, size: [.02], contact: 1 }
 shape: ssCylinder, size: [.1, .3, .02], contact: 1 }

Include: <../../rai-robotModels/scenarios/panda_fixGripper.g>

Edit panda_base (base): { Q: "t(0 0 .05) d(90 0 0 1)" }