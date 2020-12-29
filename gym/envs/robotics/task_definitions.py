

constraints = dict(sideways_1=[dict(origin="S0_0", target="S0_0", distance=0.03),
                               dict(origin="S0_8", target="S0_8", distance=0.03),
                               dict(origin="S8_0", target="S0_0",
                                    distance=0.03, noise_directions=(1, 0, 0)),
                               dict(origin="S8_8", target="S0_8", distance=0.03, noise_directions=(1, 0, 0))],

                   sideways_2=[dict(origin="S0_0", target="S0_0", distance=0.03),
                               dict(origin="S0_8", target="S0_8", distance=0.03),
                               dict(origin="S8_0", target="S0_0", distance=0.03),
                               dict(origin="S8_8", target="S0_8", distance=0.03), ],

                   diagonal_1=[dict(origin="S8_0", target="S0_8", distance=0.03, noise_directions=(1, -1, 0)),
                               dict(origin="S0_8", target="S0_8", distance=0.03), ])
