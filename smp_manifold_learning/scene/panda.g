Include: <../rai-robotModels/scenarios/panda_fixGripper.g>
#box (world): { shape:ssBox, Q: [.3, .3, 1], size: [.1,.1,.1,.005], color: [1 .5 0], contact: 0 }

#goal (world): { shape:ssBox, Q: [-0.3, .3, 1], size: [.1,.1,.1,.005], color: [.5], contact: 0 }

#obs (world): { shape:ssBox, Q: [0, .3, 1], size: [.1,.4, .4,.005], color: [1 0 0], contact: 1 }