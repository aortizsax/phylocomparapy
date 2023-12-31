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
#
# 

import dendropy
import numpy as np
from scipy.stats import norm

def brownian_motion(n, dt, sigma):
    """
    Generates a Brownian motion process.

    Args:
        n (int): The number of steps.
        dt (float): The time increment between steps.
        sigma (float): The standard deviation of the increments.

    Returns:
        np.ndarray: An array of Brownian motion values.
    """
    # Calculate the square root of the time increment
    sqrt_dt = np.sqrt(dt)

    # Generate an array of random increments from a normal distribution
    increments = norm.rvs(scale=sigma * sqrt_dt, size=n)

    # Calculate the cumulative sum of the increments
    bm_values = np.cumsum(increments)

    return bm_values
    


def brownian_motion_phylogeny(tree, sigma=1.0, root_value=0.0):
    """
    Generates a Brownian motion process fitted to a given phylogenetic tree.

    Args:
        tree (dendropy.Tree): The phylogenetic tree object.
        sigma (float): The standard deviation of the Brownian motion increments.
        root_value (float): The initial value of the Brownian motion process at the root.

    Returns:
        dict: A dictionary mapping each tree node to its corresponding Brownian motion value.
    """
    # Initialize the dictionary to store Brownian motion values
    bm_values = {}

    # Set the root value
    bm_values[tree.seed_node] = root_value

    # Traverse the tree in postorder
    for node in tree.postorder_node_iter(exclude_seed_node=True):
        # Get the parent node and its corresponding Brownian motion value
        parent_node = node.parent_node
        parent_value = bm_values[parent_node]

        # Calculate the time difference between the parent and current node
        time_diff = node.edge_length

        # Generate a random increment from a normal distribution
        increment = norm.rvs(scale=sigma * np.sqrt(time_diff))

        # Calculate the Brownian motion value at the current node
        bm_value = parent_value + increment

        # Store the Brownian motion value for the current node
        bm_values[node] = bm_value

    return bm_values


