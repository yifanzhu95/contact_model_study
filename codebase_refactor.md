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

This file should contain three classes. One should be a TaskBase, a TaskRole, and a TaskBaseConfig.


**Class TaskRole**

Just an enum to to indicated wither a task is a rollout or an eval task.

**Class TaskBaseConfig**

This should be a config for each task which stroes the values which can change between task instances. This shoould cotain things like time step, the default path, seeds, ect. 

**Class TaskBase**

Just a base class for other task types.

*function* init() $\rightarrow$ TaskBase: instanitiates the task and should take in any parameters needed.

*function* getModelPath() $\rightarrow$ string: Returns the path to the tasks MJCF model.

*function* calcCosts(VectorizedSimulator) $\rightarrow$ Numpy Array: Takes in a vectorized simulator and gets the required state, actuation and other values needed to calculate the cost for each world. if subclasses need more infromation to avoid more calculations expand the signature. It should not reproduce the functionality on the CPU.

*function* isFailure(Simulator) $\rightarrow$ Boolean: Should take in a simulator and return weather the simulator has failed. If its a vecotrized simulator it should be a array of booleans and should be capturaable. 

*function* isSuccess(Simulator) $\rightarrow$ Boolean: Should take in a simulator and return weather the simulator has succeded. If its a vecotrized simulator it should be a array of booleans and should be capturaable. 

*function* alignRendererConfigWithTask(RendererConfig) $\rightarrow$ None: takes in a renderer config then assigns all of the parameters it needs to. 

*function* setSimToInitialState(Simulator) $\rightarrow$ None: Takes in a simulator and sets the simulator to the inital state. If a vecotrized simulator it should call BroadcastState function should be capturable.

*function* sampleNewGoal() $\rightarrow$ Numpy Array: Samples a new goal and returns it as a numpy array.

*function* setGoal(goal) $\rightarrow$ None: sets the tasks goal to a goal.

*function* setRendererToGoal(Renderer) $\rightarrow$ None: Takes in a Renderer and sets it to match the current goal.


**Class LeapReorient**

Just a base class for leap reorient task types.

**Class CubeReorient**

Specific, insance of LeapReorient for the cube reorient task.

**Class DuckReorient**

Specific, insance of LeapReorient for the duck reorient task.

**Class BallReorient**

Specific, insance of LeapReorient for the ball reorient task.

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

This directory contains the code to create different renderers.

### RenderBase.py

This file should contain a renderer base class, a RandererBaseConfig, a VideoRendererBase, and a VideoRendererBaseConfig.

**Class RendererBase**

This class is a base renderer.

*function* init(Task,RendererBaseConfig = None) $\rightarrow$ RendererBase: instanitiates the renderer and takes any needed parameters from the config.

*function* RenderState(q) $\rightarrow$ None: Takes in the current state and renders the current image.

*function* Close() $\rightarrow$ None: ends the current render.

**Class RendererBaseConfig**

This class is a basic class to hold the data needed to define a renderer. It should have the follwing attributes

*Attribute* width = 640
*Attribute* hight = 480
*Attribute* fps = 30.0, max fps for interactive viewer constraint for video viewers.
*Attribute* cam_name = str | None, should be none of a string. If none shoudl default to the XMLs default otherwise should use the named camera.
*Attribute* cam_pos = None | NdArray, should be set either to none or should be a Numpy Array which describes to camera postions, if set to none it should use the XMLs default for the chosen camera
*Attribute* cam_quat = None | NdArray, should be set either to none or should be a Numpy Array which describes to camera rotation as a quaterion, if set to none it should use the XMLs default for the chosen camera.
*Attribute* cam_fovy = None | float, should be set either to none or should be a float which describes to camera's vertical feild of view, if set to none it should use the XMLs default for the chosen camera.

**Class VideoRendererBase**

This class is a base video renderer and should inherent from RendererBase.

*function* getStepsPerFrame() $\rightarrow$ int: once instaiated should this function should refrence the task time step and calculate the number of steps between frames to best respect the desired fps.

*function* Save(path) $\rightarrow$ None: saves the current video.

*function* Reset() $\rightarrow$ None: clears the current video.

**Class VideoRendererBaseConfig**

This class is a basic class to hold the data needed to define a video renderer and should extend . It should add the follwing attributes on top of the one defined 

*Attribute* output_path = string, this should just be the path to save the video to if save or close is invoked without a path.



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

### ContactModelPresets.py

This script should contain the parameters for the different contact models used for rollouts; M1, M2, M3, M4. It has one function and the herder should have all of the parameters for those models. 

*function* GetContactModelSim(Name) $\rightarrow$ Simulator: This is a helper function to generate the different models. It should take in a string representing the name of ther desirned ocntact model and return a simulator which uses that model. 

### EpisodeRecorder.py

This should a class which handels all of the recording and replaying for a episode.

**Class EpisodeRecorder**

This class handels the recording for a batch of episodes. This includes recording the Configs used for each episode, the states visited, the actions selected, the successes, the failures, the planning time, ect.

*function* init(Task,Simulator,SamplingBasedPlanner,Other Parms ...) $\rightarrow$ EpisodeRecorder: This constructor should take in a refrence to the eval task, the eval simulator, and the planner. It will then grab refs to all of the objects ciritcal to the episode.

*functiion* recordStateAndAction(q,q_dot,U,simga_U = None,planing_time=None) $\rightarrow$ none: This function takes in a state and a action (and optionally, the uncertaininty associated with that action and the planning time) and adds it to the recording buffer.

*functiion* episodeFinished(finish_reason) $\rightarrow$ none: This function takes in a reason for the episode ending and ends the episode.

*function* Save(Path) $\rightarrow$ none: This function simply saves the episodes and summery of the episodes as a JSON file. The states, actions, and uncertaininty should be saved in a single npy file for each episode and are given an unique ID.

*function* Clear() $\rightarrow$ none: Clears the buffer

*function* Combine(EpisodeRecorder) $\rightarrow$ EpisodeRecorder: Combines another epiosde recorder with itself and returns it as a new episode recorder. It checks that all of the configs are the same and requieres that both are not in the process of recording a active episode.

**Class EpisodeReplayer**

This class handels the replaying for a batch of episodes. It should allow for iterating over all of states and all of the episodes. 

*function* init(path) $\rightarrow$ EpisodeReplayer: This constructor takes in a path to a record batch of episodes.


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