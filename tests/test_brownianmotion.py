#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
## Copyright (c) 2023 Adrian Ortiz-Velez.
## All rights reserved.
## 
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
## 
##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.
##     * Redistributions in binary form must reproduce the above copyright
##       notice, this list of conditions and the following disclaimer in the
##       documentation and/or other materials provided with the distribution.
##     * The names of its contributors may not be used to endorse or promote
##       products derived from this software without specific prior written
##       permission.
## 
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
## ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
## WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL ADRIAN ORTIZ-VELEZ BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
## BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
## OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
## 
##############################################################################

import unittest
if __name__ == "__main__":
    import _pathmap
else:
    from . import _pathmap

from phylocomparapy.simulate.stochastic import brownian_motion
import unittest
import numpy as np
from scipy.stats import norm
from numpy.random import Generator, PCG64
from scipy.stats import binom

class BrownianMotionTest(unittest.TestCase):
    
    def test_brownian_motion(self):
        n_steps = 1000
        time_increment = 0.01
        standard_deviation = 1.0
        n_simulations = 1000
        
        # Generate the Brownian motion process
        seed = 2345678901
        numpy_randomGen = Generator(PCG64(seed))
        scipy_randomGen = norm
        scipy_randomGen.random_state=numpy_randomGen
        
        bm_values = brownian_motion(n_steps, time_increment, standard_deviation)
        
        # Calculate the expected values using the analytical solution
        numpy_randomGen = Generator(PCG64(seed))
        scipy_randomGen = norm
        scipy_randomGen.random_state=numpy_randomGen
        expected_values = norm.rvs(scale=np.sqrt(time_increment) * standard_deviation, 
                          size=n_steps)
        expected_values = np.cumsum(expected_values)
        
        # Compare the generated values with the expected values
        self.assertTrue(np.allclose(bm_values, expected_values))
        
        
    def test_brownian_motion_var(self):
        n_steps = 1000
        time_increment = 0.01
        standard_deviation = 1.0
        n_simulations = 3000
        
        # Generate the Brownian motion process
        seed = 2345678901
        numpy_randomGen = Generator(PCG64(seed))
        scipy_randomGen = norm
        scipy_randomGen.random_state=numpy_randomGen
        
        #repeat 1000 times get varience 
        bm_simulation_values = []
        for i in range(n_simulations):
            bm_values = brownian_motion(n_steps, time_increment, standard_deviation)
            bm_simulation_values.append(bm_values[-1])
            
        bm_simulation_var = np.var(bm_simulation_values)
        expected_value = n_steps * time_increment * standard_deviation * standard_deviation

        self.assertTrue(np.allclose(bm_simulation_var, expected_value, rtol = 0.05, atol = 2))
        
if __name__ == '__main__':
    unittest.main()


if __name__ == "__main__":
    unittest.main()
