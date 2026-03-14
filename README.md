# Pablo QArm Image Drawing Application

## Overview

This project is a modular Python application that drives a Quanser QArm to draw a path extracted from an image file located in the same folder as the scripts. The system is organized as a small set of focused modules and a finite state machine controller.

At a high level the runtime loop looks like this

1. `main.py` creates the robot IO wrapper, kinematics engine, vision processor, UI, and the FSM controller
2. a fixed rate loop runs at 50 Hz
3. the UI thread pushes commands as events into a queue
4. each loop tick drains the queue and passes events to the controller
5. the controller runs one step of the current state every tick

This design keeps the robot update loop non blocking and makes it easy to extend or debug.

---

## Module structure

### `main.py`
Responsible for wiring everything together and running the fixed rate loop.

Key responsibilities
- define the drawing plane in the robot base frame
- instantiate `QArmIO`, `KinematicsEngine`, `VisionProcessor`, `TerminalUI`, and `RobotController`
- run the main loop at a constant rate using `TrajectoryTimer`

### `ui_handler.py`
A terminal based interface implemented as a background thread.

Key responsibilities
- read commands without blocking the robot loop
- convert text commands into typed `UIEvent` messages
- push events into a thread safe queue for the controller

Supported commands
- `load <filename>` load an image from the same folder
- `preview` live preview of interpreted path, no robot motion
- `start` execute drawing
- `stop` emergency stop
- `reset` exit emergency stop and return to idle
- `home` move to home pose
- `grip open|close` open or close gripper
- `rotate <deg>` set wrist angle absolute
- `quit` exit the program

### `vision_processor.py`
Converts an image into a 2D drawing path.

Key responsibilities
- load an image from disk
- extract a contour from the image
- scale and center the contour in meters inside the drawing area
- output a list of planar waypoints with pen up and pen down transitions

Implementation notes
- OpenCV is used if available
  - grayscale
  - Otsu threshold
  - external contours
  - pick largest contour by area
- fallback uses Pillow
  - Otsu threshold
  - binary edge detection
  - angle ordering around centroid to produce an ordered loop

The output of this module is a list of `(x, y, pen_down)` points in meters in a drawing plane coordinate system.

### `kinematics_engine.py`
Maps the 2D path into the robot base frame and produces robot joint commands.

Key responsibilities
- define a `DrawingPlane` transform from plane coordinates into the robot base frame
- resample the path so step size is bounded for smooth motion
- insert Z transitions for pen up and pen down
- call Quanser IK through `QArmUtilities.qarm_inverse_kinematics`

Important detail
- IK uses a wrist orientation parameter called `gamma`
- the controller uses the most recently set wrist angle from the `rotate` command as `gamma` during homing and drawing
- this ensures the drawing is executed with the wrist angle the user chose

### `robot_controller.py`
Implements the finite state machine and contains the real time logic.

Key responsibilities
- consume `UIEvent` events from `ui_handler.py`
- transition between states
- send robot joint and gripper commands continuously
- generate and cache drawing jobs from the vision processor and kinematics engine
- implement preview, homing, drawing, gripper control, wrist rotation, and emergency stop
- implement Z compliance behavior to avoid pushing down too hard when the pen touches the surface

---

## Core design idea

### Non blocking control loop

The controller is stepped at a fixed rate. The UI input runs in a separate thread and only enqueues events. The robot loop stays deterministic because it never blocks on terminal input or on a blocking matplotlib call.

---

## Finite state machine

### States

- **IDLE**
  - holds the last commanded pose and waits for commands
- **PREVIEW**
  - shows a live animated preview of the interpreted path
  - does not move the robot
  - keeps sending hold commands so the arm stays stiff
- **CALIBRATING**
  - validates that an image has been loaded
  - builds the drawing job if needed
  - transitions to homing before drawing
- **HOMING**
  - moves the robot to a safe home pose using IK
  - uses the stored wrist angle as the IK gamma so wrist orientation stays consistent
- **DRAWING**
  - executes waypoint by waypoint IK commands
  - uses the stored wrist gamma from `rotate`
  - includes Z compliance logic when pen is down
- **GRIP_OPENING**
  - opens the gripper while holding joint pose
- **GRIP_CLOSING**
  - closes the gripper using a current limited ramp
- **ROTATE_EE**
  - rotates the wrist to an absolute target angle
  - updates the stored wrist gamma when done
- **EMERGENCY_STOP**
  - freezes motion by holding the last commanded pose
  - only `reset` returns to IDLE

---

## Crucial implementation details

### 1. Holding the last commanded joint pose

A key stability technique is that the controller always keeps a persistent joint target vector

- `self._last_phi_cmd`

and repeatedly sends it while in IDLE and in other non motion states.

This prevents the arm from drooping due to missing commands or due to accidentally replacing the hold target with the measured pose.

### 2. Live preview without robot motion

The preview state uses matplotlib `FuncAnimation` with `plt.show(block=False)` and `plt.pause(...)` to pump GUI events.

Important behaviors
- the robot continues to receive hold commands during preview
- preview only runs when in PREVIEW state
- when the preview window is closed the controller transitions back to IDLE

### 3. Wrist angle drives the drawing

The `rotate <deg>` command sets a wrist target.

The controller performs a smooth wrist motion in small increments and once it reaches the target

- it stores the final wrist joint angle into `self._wrist_gamma_rad`

During homing and drawing, every IK call uses

- `gamma_rad = self._wrist_gamma_rad`

This means the drawing is executed with the wrist angle you set earlier.

### 4. Gripper close current limiting

Closing the gripper is implemented as a ramp rather than a single jump.

Behavior
- increase gripper command by `GRIP_STEP_PER_TICK` up to `GRIP_MAX_CLOSE_CMD`
- monitor gripper current
- if current stays above a threshold for several ticks or the close times out
  - back off slightly
  - return to IDLE

This reduces persistent overload faults while still providing a usable close behavior.

### 5. Z compliance that behaves like impedance at the pen tip

True impedance control would require force sensing and torque control. The QArm is position commanded, so the project implements an approximation.

The idea
- use joint current rise as a proxy for contact force at the end effector
- adjust Z so the pen maintains light contact without digging into the surface

Two phase behavior when the waypoint is pen down

**Touch down phase**
- approach from pen up Z
- descend in small steps until contact is detected
- contact detection uses a current rise threshold
- store the contact Z as `contact_z_m`

**Compliant draw phase**
- while moving along XY
- compute current rise relative to baseline
- adjust Z proportionally using an admittance gain
- clamp Z to a small band around the contact height
- if contact is lost, fall back to touch down again

Net effect
- the robot touches the surface
- it stops pushing further downward
- it begins moving along the path while maintaining gentle contact

---

## Typical usage flow

1. start the program
2. load an image file
3. rotate wrist to a preferred drawing angle
4. preview the interpreted path
5. start drawing

Example

- `load circle.png`
- `rotate 90`
- `preview`
- close the preview window
- `start`

---

## Tuning notes

### Z compliance tuning

If the pen presses too hard
- increase contact threshold
- increase admittance gain
- reduce allowable Z below contact

If the pen never detects contact
- decrease contact threshold
- increase descent step slightly

If the contact metric is noisy
- lower the current filter alpha

### Gripper tuning

If you still see overload warnings
- lower `GRIP_CURRENT_LIMIT_A`
- lower `GRIP_MAX_CLOSE_CMD`
- increase `GRIP_BACKOFF` slightly

If gripping is too weak
- increase `GRIP_CURRENT_LIMIT_A` slightly
- reduce `GRIP_BACKOFF` slightly