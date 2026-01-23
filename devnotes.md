
# Development notes

## Desired capabilities

### Must have

- [ ] Run start-to-finish from command line with input file
- [ ] Import as object and run in a script or notebook
- [ ] When run as object, have function to run the whole thing start to finish
- [ ] When run as object, also allow "piecewise" initialization, run for a period of time, query outputs/fields, run again for an additional period of time, etc.
- [ ] Select a raster or hex grid
- [ ] When imported into Python, have option to pass parameters as a dict (which allows having some parameters be spatially varying)
- [ ] Output to NetCDF (raster) and VTK (hex or raster) with user control on fields to be included

### Nice to have

- [ ] Option to read spatially varying initial conditions from some kind of input file
- [ ] Restart capability: option to restart from a previous run, preserving all fields, parameters, etc.
- [ ] Have a collection of basic tutorials
- [ ] Include the examples from the All About EGBE paper

### Icing on the cake 

- [ ] Work with a global grid
- [ ] Have a built-in mechanism to configure initial conditions (e.g., specify one or more rectangular areas that have certain grain-size, gamma, or lithology data)
- [ ] Have some kind of a visualization tool (e.g., in paraview?) that displays stream networks as line segments
