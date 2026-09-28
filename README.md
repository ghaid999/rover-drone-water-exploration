# AI-Assisted Rover–Drone Collaboration for Water-Potential Target Selection

An autonomous rover and drone working as a team to decide **where to look for
water** on a simulated planetary surface, using a machine-learning model trained
on real Earth groundwater data.

---

## What this project does

Finding water is a central goal of planetary exploration, but a surface robot
cannot see underground. Without dedicated drilling or sounding instruments, a
rover has no direct way to tell whether water lies beneath a given patch of
ground.

This project takes a different approach. Instead of trying to detect water
directly, the system **ranks locations by how likely they are to hold water**,
using the same environmental signals that hydrogeologists use on Earth: rainfall,
elevation, slope, radar backscatter, drainage patterns and geological fractures.
The robot then spends its limited time and energy on the most promising
locations rather than searching blindly.

The rover leads the mission. It sweeps its surroundings, deploys a drone to
survey candidate locations from the air, collects a water-potential score for
each one, ranks them, and drives to the best.

---

## How it works

![System workflow](docs/figures/system_flow_simple.png)

**1. A model is trained offline, on real data.**
A Random Forest classifier learns from 40 field-surveyed groundwater wells in the
Shalatein region of Egypt, an arid area used here as an Earth analogue for Mars.
Each well is paired with nine geospatial measurements taken at that location.
The model outputs a probability rather than a yes/no answer, which is what makes
ranking possible.

**2. The simulated world is connected to real geography.**
This is the central problem the project had to solve. The Gazebo world is a
fictional 50 x 50 m patch of terrain. It contains no satellite imagery, no
rainfall records, nothing the model was trained on. So a coordinate mapping
translates any simulated position into a real geographic coordinate inside the
Shalatein study area, where the nine raster layers can actually be sampled.

A simulated rover therefore queries genuine satellite and terrain data, without
the simulation needing to contain any of it.

**3. The mission runs as a loop.**
The rover sweeps the area in five lanes. Once coverage is complete it deploys the
drone, which surveys 20 candidate locations one at a time. For each, the drone
retrieves the nine features through the coordinate mapping, scores them with the
trained model, and returns the result. The rover ranks all 20 and navigates to
the highest-scoring location.

---

## Understanding the code

If you are reading this repository to understand the system rather than to run
it, this order will make the most sense.

**Start here — `ai_model/feature_generator.py`**
The idea the project rests on. It maps a simulated coordinate to a real
geographic one and samples nine raster layers there. The margin constant is worth
understanding: without it, roughly 11 % of sampled points landed on gaps in the
real raster coverage, because the published survey boundary does not exactly
match where satellite data actually exists.

**Then — `ai_model/water_potential_scorer.py`**
Turns those nine features into a single score between 0 and 1. Short file, but
note why it calls `predict_proba()` instead of `predict()`: a binary
water/no-water label cannot be ranked, and ranking is the entire point.

**Then — `ros2_ws/src/drone_control/drone_control/drone_node.py`**
The survey side. Receives a location, retrieves features, scores them, returns
the result. The drone is modelled kinematically: it does not physically fly to
the survey point, and the file says so where it matters.

**Then — `ros2_ws/src/rover_control/rover_control/rover_node.py`**
The largest file, and the mission logic. A state machine moving through
exploring, surveying, investigating and done. Worth reading for how the drone
deployment decision is made and how candidates get ranked.

**Finally — the evaluation scripts in `data/groundwater_dataset/`**
Not needed to run the system, but they contain the experiments behind the
reported results: the comparison against published methods, the target-selection
accuracy, and the component ablation.

### How the pieces talk to each other

The two robots are separate ROS 2 nodes that communicate over topics. A bridge
relays messages between ROS 2 and Gazebo, which use different messaging systems.

```
rover_node  --/survey_request-------->  drone_node
drone_node  --/drone_survey_result-->   rover_node

rover_node  --/cmd_vel-------------->   Gazebo      (via bridge)
drone_node  --/drone_cmd_vel-------->   Gazebo      (via bridge)
Gazebo      --/odometry------------->   rover_node  (via bridge)
Gazebo      --/clock---------------->   both nodes  (via bridge)
```

`/odometry` is how the rover knows where it is, and `/clock` is how both nodes
keep time. If either is missing from the bridge, the system fails quietly rather
than loudly, so both are worth checking first when something does not work.

---

## Requirements

| Component | Version used |
|---|---|
| Ubuntu | 26.04 |
| ROS 2 | Lyrical Luth |
| Gazebo | Jetty 10.5.0 |
| Python | 3.14 |

```bash
sudo apt install ros-lyrical-desktop ros-lyrical-ros-gz
pip install -r requirements.txt --break-system-packages
```

## Dataset

The raw geospatial dataset is roughly 4.6 GB and is **not** included here. It
accompanies a 2026 *Scientific Reports* publication on groundwater potential
mapping in the Shalatein region: https://zenodo.org/records/21502024

Extract it so the study-area folder sits at
`data/groundwater_dataset/Shalatein Paper/`.

The processed feature table and the trained model **are** included, so a mission
can be run without downloading the raw data. The raw dataset is only needed to
re-train the model or re-run the evaluation scripts.

## Setup

```bash
git clone <repository-url>
cd rover_drone_project
pip install -r requirements.txt --break-system-packages
cd ros2_ws && colcon build && source install/setup.bash
```

Add the simulation models to Gazebo's search path (append to `~/.bashrc`):

```bash
export GZ_SIM_RESOURCE_PATH=~/rover_drone_project/simulation/models:$GZ_SIM_RESOURCE_PATH
```

If the Gazebo window fails to render, force software rendering:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

## Running a mission

Five terminals, started in this order. Source the workspace
(`source ~/rover_drone_project/ros2_ws/install/setup.bash`) in terminals 3 to 5.

**1 — Simulation server**
```bash
cd simulation/worlds && gz sim -s -r rover_drone_world.sdf
```

**2 — Graphical view**
```bash
gz sim -g
```

**3 — ROS 2 to Gazebo bridge**
```bash
./scripts/run_bridge.sh
```

**4 — Drone node** (wait for `Drone node ready`)
```bash
ros2 run drone_control drone_node --ros-args -p use_sim_time:=true
```

**5 — Rover node**
```bash
ros2 run rover_control rover_node --ros-args -p use_sim_time:=true
```

A full mission takes roughly 15 minutes of wall-clock time, as the simulation
runs below real time.

> **The `/clock` topic must be bridged.** With `use_sim_time:=true` and no
> `/clock`, a node's timer never fires and the rover sits still with no error
> message. `scripts/run_bridge.sh` includes it. To confirm before starting the
> rover: `ros2 topic echo /clock --once`

### Expected output

```
Rover mission started. Sweeping 10 waypoints ...
Sweep waypoint 1/10 reached ...
...
Exploration ended: sweep pattern completed ... -- deploying drone.
Generated 20 candidate survey points.
Requesting survey 1/20 at ...
Survey result: (...) -> score=...
Ranking complete over 20 candidates. Best: ...
Mission complete (arrived). Final position ...
```

## Reproducing the evaluations

These require the raw dataset. Run from `data/groundwater_dataset/`:

```bash
python3 build_feature_table.py        # build the labelled training table
python3 train_model.py                # train and cross-validate
python3 compare_published_methods.py  # compare against published WOE/EBF maps
python3 target_selection.py           # selection accuracy vs. baselines
python3 selection_diagnostics.py      # single-feature heuristics, candidate-set size
python3 ablation.py                   # component ablation
```

## Repository structure

```
ai_model/                       sim-to-real mapping, scorer, trained model
data/                           ground truth and analysis scripts
simulation/                     Gazebo world, rover, drone, terrain
ros2_ws/src/rover_control/      rover node, mission leader
ros2_ws/src/drone_control/      drone node, survey and scoring
scripts/run_bridge.sh           ROS 2 to Gazebo topic bridge
docs/                           report and figures
```

## Results

| Measure | Value |
|---|---|
| Model accuracy (5-fold CV) | 75.8 % ± 9.6 % |
| AUC on held-out wells, local | 0.847 |
| AUC on held-out wells, regional | 0.878 |
| Mean distance to nearest held-out well | 6.75 km (11.40 km random) |
| Rover traverse vs. rover-only | 12.50 km vs. 40.93 km |

Full results, including confidence intervals and limitations, are in `docs/`.

## Attribution

The Mars terrain mesh is taken from
[spaceros_gz_demos](https://github.com/david-dorf/spaceros_gz_demos), developed
for NASA's Space ROS Sim Summer Sprint Challenge; see that repository for its
licence.

The groundwater dataset is from the Zenodo record above; please cite the
accompanying publication if you use it.

## Licence

MIT, see `LICENSE`. Third-party assets keep their own terms.