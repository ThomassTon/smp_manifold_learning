Include: <pandaDual_parallel.g>

#box (world): { shape:ssBox, Q: [.3, .1, .8], size: [.3,.1,.02,.005], color: [1 .5 0], contact: 1 }

#goal (world): { shape:ssBox, Q: [.3, .2, 1.2, -0.946902, -0.00536831, -0.0133451, -0.3212], size: [.3,.1,.02,.005], color: [.5], contact: 0 }
#goal (world): { shape:ssBox, Q: [.1, .2, 1.4, 0.845, 0.488, -0.108, 0.187], size: [.3,.1,.02,.005], color: [.5], contact: 0 } # z and x
# goal (world): { shape:ssBox, Q: [.1, .2, 1.4, 0.991, 0.0, 0.131, 0.0], size: [.3,.1,.02,.005], color: [.5], contact: 0 }  # y
#goal (world): { shape:ssBox, Q: [.1, .2, 1.4, 0.653, 0.653, 0.271, 0.271], size: [.3,.1,.02,.005], color: [.5], contact: 0 }  # y and z

#obs (world): { shape:ssBox, Q: [.3, .15, 1], size: [.3,.2, .01,.005], color: [1 0 0], contact: 1 }

sphere (world): { shape:sphere, Q: [.3, .1, 1], size: [.2], color: [1 .5 0], contact: 0 }




