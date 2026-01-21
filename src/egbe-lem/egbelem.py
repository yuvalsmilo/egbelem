#! /usr/bin/env python
# coding: utf-8
"""
EGBE-LEM: Landscape Evolution Model that implements the theory behind
the ExtendedGravelBedrockEroder.
"""

import sys

import numpy as np
from landlab import RasterModelGrid, HexModelGrid
from landlab.core import load_params
from landlab.components import ExtendedGravelBedrockEroder, PriorityFloodFlowRouter, FlowAccumulator

from model_base import LandlabModel


class EgbeLem(LandlabModel):
    """Landscape Evolution Model using fluvial EGBE theory."""

    DEFAULT_PARAMS = {
        "grid": {
            "source": "create",
            "create_grid": {
                "RasterModelGrid": [
                    (31, 31),
                    {"xy_spacing": 1000.0},
                ],
            },
        },
        "clock": {"start": 0.0, "stop": 10000.0, "step": 10.0},
        "output": {
            "plot_times": 2000.0,  # float or list
            "save_times": 10000.0,  # float or list
            "report_times": 1000.0,  # float or list
            "save_path": "bigantr_run",
            "clobber": True,
            "fields": None,
            "plot_to_file": True,
        },
        "initial_conditions": {
            "initial_sed_thickness": 1.0,
            "random_topo_amp": 10.0,
            "initial_topo": 0.0,
        },
        "baselevel": {
            "uplift_rate": 0.0001,
        },
        "flow_routing": {
            "flow_metric": "D8",
            "update_flow_depressions": True,
            "depression_handler": "fill",
            "epsilon": True,
            "bankfull_runoff_rate": 1.0,
        },
        "fluvial": {
            "intermittency_factor": 0.01,
            "transport_coefficient": 0.041,
            "sediment_porosity": 1.0 / 3.0,
            "depth_decay_scale": 1.0,
            "plucking_coefficient": 1.0e-4,
            "number_of_sediment_classes": 1,
            "init_fraction_per_class": [1.0],
            "abrasion_coefficient": [1.0e-4],
            "coarse_fraction_from_plucking": [0.5],
            #"rock_abrasion_index": 0,
        },
    }

    def __init__(self, params={}):
        """Initialize the model."""
        super().__init__(params)

        # Set up grid fields
        ic_params = params["initial_conditions"]
        if not ("topographic__elevation" in self.grid.at_node.keys()):
            self.grid.add_zeros("topographic__elevation", at="node")
            self.grid.at_node["topographic__elevation"][
                self.grid.core_nodes
            ] += ic_params["random_topo_amp"] * np.random.rand(
                self.grid.number_of_core_nodes
            )
        if not ("soil__depth" in self.grid.at_node.keys()):
            self.grid.add_zeros("soil__depth", at="node")
            self.grid.at_node["soil__depth"][:] = ic_params["initial_sed_thickness"]
        if not ("bedrock__elevation" in self.grid.at_node.keys()):
            self.grid.add_zeros("bedrock__elevation", at="node")
            self.grid.at_node["bedrock__elevation"][:] = self.grid.at_node["topographic__elevation"] - self.grid.at_node["soil__depth"]
        self.topo = self.grid.at_node["topographic__elevation"]
        self.sed = self.grid.at_node["soil__depth"]
        self.rock = self.grid.at_node["bedrock__elevation"]
        self.topo[self.grid.core_nodes] = ic_params["initial_topo"]
        self.rock[self.grid.core_nodes] = self.topo[self.grid.core_nodes] - self.sed[self.grid.core_nodes]

        # Store parameters
        self.uplift_rate = params["baselevel"]["uplift_rate"]

        # Instantiate and initialize components: flow router
        flow_params = params["flow_routing"]
        if isinstance(self.grid, RasterModelGrid):
            self.router = PriorityFloodFlowRouter(
                self.grid,
                surface="topographic__elevation",
                flow_metric=flow_params["flow_metric"],
                update_flow_depressions=flow_params["update_flow_depressions"],
                depression_handler=flow_params["depression_handler"],
                epsilon=flow_params["epsilon"],
                accumulate_flow=True,
                runoff_rate=flow_params["bankfull_runoff_rate"]
            )
        else:
            self.router = FlowAccumulator(
                self.grid,
                runoff_rate=flow_params["bankfull_runoff_rate"],
                depression_finder='DepressionFinderAndRouter'
            )
        print("BFR", flow_params["bankfull_runoff_rate"])

        # Instantiate and initialize components: fluvial transport, erosion, deposition
        gbe_params = params["fluvial"]
        self.eroder = ExtendedGravelBedrockEroder(
            self.grid,
            intermittency_factor=gbe_params["intermittency_factor"],
            transport_coefficient=gbe_params["transport_coefficient"],
            sediment_porosity=gbe_params["sediment_porosity"],
            depth_decay_scale=gbe_params["depth_decay_scale"],
            plucking_coefficient=gbe_params["plucking_coefficient"],
            number_of_sediment_classes=gbe_params["number_of_sediment_classes"],
            init_fraction_per_class=gbe_params["init_fraction_per_class"],
            abrasion_coefficients=gbe_params["abrasion_coefficients"],
            coarse_fractions_from_plucking=gbe_params["coarse_fractions_from_plucking"],
            rock_abrasion_index=gbe_params["rock_abrasion_index"],
        )

    def update(self, dt):
        """Advance the model by one time step of duration dt."""
        #print("BL Update here", self.current_time, dt, self.uplift_rate)
        #print(" topo before uplift", self.topo[self.grid.core_nodes])
        dz = self.uplift_rate * dt
        self.topo[self.grid.core_nodes] += dz
        self.rock[self.grid.core_nodes] += dz
        #print(" topo after uplift but before GBE", self.topo[self.grid.core_nodes])
        self.router.run_one_step()
        self.eroder.run_one_step(dt)
        #print(" topo after GBE", self.topo[self.grid.core_nodes])
        self.current_time += dt


if __name__ == "__main__":
    if len(sys.argv) > 1:
        params = load_params(sys.argv[1])
    else:
        params = {}
    bigantr = BigantrLEM(params)
    bigantr.run()
