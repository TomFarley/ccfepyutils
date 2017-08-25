# CCFE plotting class specification and plan 
Author: Tom Farley

Date: 08-2017

## Purpose
General purpose plotting class for easy production of publication ready
figures.

### Core goals
- Remove repetition from plot creation
- Provide easy access to complex/fiddly features
- Be easily accessible to existing users of matplotlib
- Maintain easy access to all matplotlib functionality
- Be compatible with fusion specific functionality, meta data etc.
  without containing application specific code
- Provide a unified interface with common plotting functionality
  for a wide range of applications
  - New funtionality is automatically available across wide codebase

### Suplementary goals
- Easy access record of data in figure
- Provide provenence tools for published figures
- Provide single interface to multiple plotting libraries: matplotlib,
  plotly

## Scope

### Use cases
- Interactive use in python console
- Plotting data in scripts
- Providing unified plotting backend for other applications/classes

### Users
#### Primary users
- Include in ccfetools package
  - Aim for widespread use on culham site

#### Secondary users
- Collaborators using CCFE codes

## Philosophy
- Simplicity - KISS
- Handle absence of dependencies cleanly
- Most methods accept external axes for indirect/external use

## Development approach
- All new code added to plotter_dev_<username> branch
- Code review required for addition to ccfetools/plotter branch

## Testing
- Provide testing demos for all funtionality

## Detailed specification and feature set

### Baseline

#### Figure
- Matplotlib rc files
- Saved image output
- Nice legends
- Easy best fits to data (linear, exponential, misc fits with stats)
- Easy plotting of functions (no need to generate x input)

#### 1D plots
- Line color colormaps
- Multiple x/y axes

#### 2D plots
- Images: imshow
- Conour maps: contourf w/ w/o contour lines
- Nice colorbars: limits

#### 3D plots
- 3D scatter
- Surface plots

### Secondary
- Annotated h and v lines
- 2D line width/color dependence on z parameter (port seg_line_plot)
- Arrow plots (port arrowplot)

### Additional
- Hoverinfo
- Interact with settings object for output file path
- Provide figure manager for manging multiple figures