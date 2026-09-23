Purpose:
    The codebase has become to big to maintain a refactor is needed.

New Structure:

    ContactModelStudy
    |---__init__.py
    |---Simulators
    |   |---Simulator.py
    |   |---VectorizedSimulator.py
    |   |---Mujoco.py
    |   |---VectorizedMujoco.py
    |   |---ComFree.py
    |   |---XPBD.py
    |   |---Pinocchio.py
    |   |---Drake.py
    |---Tasks
    |   |---TaskBase.py
    |   |---LeapReorient.py
    |   |---CubeReorient.py
    |   |---BallReorient.py
    |   |---DuckReorient.py
    |   |   |---XML_Files
    |   |   |   |---env_leap_eval_cube.xml
    |   |   |   |---env_leap_rollout_cube_high_high.xml
    |   |   |   |---(Put Other XML files here)
    |---Drivers
    |   |---run_episodes.py
    |   |---run_episodes_parallel.py
    |---SamplingBasedPlanners
    |   |---SamplingBasedPlannerBase.py
    |   |---MPPI.py
    |   |---CEM.py
    |   |---PS.py
    |---Renderers
    |   |---RendererBase.py
    |   |---MujocoVideoRenderer.py
    |   |---MujocoInteractiveRenderer.py
    |---Utils
    |   |---EpisodeIO.py
    Experiments
    |---HPC
    Results
    Logs
    Tests
    Videos

# Detailed Descriptions:

## Simulators:

This directory contains all the code needed to instantiate the various simulators from MJCF XML files. 

### Simulator.py:

This file should contain two classes; the Simulator class which will be the base class for all other simulators and a simulator config dataclass.

**Class Simualtor**

This class is the base class for all simulators. These classes are basic wrappers for each of the simulators and expose only the nessisary functions for the study. 

*function* init(XML_path,sim_config): This function should take in a string representing a XML file or the path to an XML file and instantiates the model and all other needed quantities and objects. This should also take in a simulation config which describes the physics params needed.

*function* Step(step=1) $\rightarrow$ None: This function steps the model forward one or more time steps (defaults to 1).

*function* SetControl(U) $\rightarrow$ None: Takes in a numpy array and simply sets the current control.

*function* GetControl() $\rightarrow$ q:Numpy Array: gets the current control input and returns it as a numpy array.

*function* SetState(Q) $\rightarrow$ None: sets Takes in a numpy array and sets the current state of the simulator.

*function* GetState() $\rightarrow$  (q:Numpy Array, q_dot:Numpy Array): Returns the full state (both the state and the velocity) of the simulator as a numpy array.



### VectorizedSimulator.py:

This file should contain two classes. Firstm, the VectorizedSimulator class which will be the base class for all of the parallized simulators and should extend/be an initance of the simulator class. It shoould also have a VectorizedSimulatorConfig data class.

**Class VectorizedSimulator**

This class is the base class for all parallel simulators. This class should extend the Simulator class. Only non-inhertited or overwirtten functions signatures will be described here.

*function* init(time_step,sim_config,N): This function should take in a string representing a XML file or the path to an XML file and make N instantiates the model and all other needed quantities and objects.

*function* SetControlSequence(U_n) $\rightarrow$ None: Takes in a numpy array and sets a sequence of H control inputs.

*function* Step_GPU(steps=1) $\rightarrow$ None: Calls the model set function on the GPU.



### Mujoco.py: 

Contains a class which is instance of a Simulator (non-vectorized) which runs standard CPU MuJoCo.

### Pinocchio.py:

Contains a class which is instance of a Simulator and runs the pinocchio simulator.

### Drake.py: 

Contains a class which is instance of a Simulator and runs the drake simulator.

### VectorizedMujoco.py:

Contains a class which is a instance of a VectorizedSimulator which runs standard MuJoCo Warp.

### ComFree.py:

Contains a class which is a instance of a VectorizedSimulator which runs ComFree Warp.

### XPBD.py:

Contains a class which is a instance of a VectorizedSimulator which runs the XPBD simulator.


## Tasks:

This directory should contain the code which define specific tasks and all of the values associated with that task.

### TaskBase.py

This file should contain two class. That class should be a empty class which is just a base class for other tasks.

**Class TaskBase**

Just a base class for other task types.

*function* init() $\rightarrow$ TaskBase: instanitiates the task and should take in any parameters needed.

*function* getModelPath() $\rightarrow$ string: Returns the path to the tasks MJCF model.

*function* calcCosts(q,q_dot,u) $\rightarrow$ Numpy Array: Takes in a numpy array of states and actuations and calculates the cost associated with that sequence of states.

*function* calcCosts_GPU(q,q_dot,u) $\rightarrow$ warp array: Takes in a warp array of states and actuations and calculates the cost associated with that sequence of states on the GPU.

**Class LeapReorient**

Just a base class for leap reorient task types.

**Class CubeReorient**

Specific, insance of LeapReorient for the cube reorient task.


## Drivers:

This directory contains the basic scripts to run episodes in the various ways

### run_episodes.py

Basic script which exposes most of the parameters as command line arguments. 

### run_episodes_parallel.py

Ignore for now.

## SamplingBasedPlanners:

This directory contains all of the code to instantiate various different smapling based planners.

### SamplingBasedPlannerBase.py

This file should contain two class. That class should be a empty class which is just a base class for other tasks.

**Class SamplingBasedPlannerBase**

Just a base class for all of the sampling based planner types.

*function* init(VectorizedSimulator,Task,Config) $\rightarrow$ SamplingBasedPlannerBase: instanitiates the planner and should take in any parameters needed. It should take in a VectorizedSimulator, a Task, and a Config object. 

*function* Plan(q,q_dot) $\rightarrow$ u_star;Numpy Array: Takes in the current state and returns the optimal next action


### MPPI.py

This file should contain two classes. First, a base MPPI class which implements a MPPI planner and a MPPI_config class which is a data class to define the planner.

**Class MPPI**

Just a base class for all of the sampling based planner types.

*function* init(VectorizedSimulator,Task,MPPI_Config) $\rightarrow$ MPPI: instanitiates the planner and should take in any parameters needed. It should take in a VectorizedSimulator, a Task, and a Config object. 

*function* Plan(q,q_dot) $\rightarrow$ Numpy Array: Takes in the current state and returns the optimal next action as a numpy array.

### CEM.py

Ignore for now

### PS.py

Ignore for now

## Renderers:

This directory contains the code to create the renderers.

### RenderBase.py

This file should contain a renderer base class and a RandererBaseConfig.

**Class RendererBase**

This class is a base renderer.

*function* init(Task,RendererBaseConfig) $\rightarrow$ RendererBase: instanitiates the renderer and takes any needed parameters from the config.

*function* RenderState(q) $\rightarrow$ None: Takes in the current state and renders the current image.

*function* Close() $\rightarrow$ None: ends the current render.

### MuJoCoVideoRenderer.py

This file should contain a renderer which simply wraps MuJoCo and uses it for the video renderer.

**Class MuJoCoVideoRenderer**

Wrapper to use the mujoco video rendering functionalities.

### MuJoCoInteractiveRenderer.py

This file should contain a renderer which simply wraps MuJoCo and uses it for its interactive viewer. This does not need to allow user inputs to effect the simulation so the simulators should not need to worry about taking inputs from the renderers.

**Class MuJoCoVideoRenderer**

Wrapper to use the mujoco viewer for rendering functionalities.

## Utils:

This directory just contains various utilities used throughout the code.

## Experiments:

Ignore for Now

## Results:

Ignore for Now

## Logs:

Ignore for Now

## Tests:

Ignore for Now

## Videos:

Ignore for Now