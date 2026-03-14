# Pablo


# Create a virtual environment
python -m venv .venv

# Activate it
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

How to tune it (robot_controller.py)
Increase CONTACT_CURRENT_RISE_A slightly (example 0.45)
Increase Z_ADMITTANCE_M_PER_A slightly (example 0.0012)
Increase Z_ABOVE_CONTACT_MAX_M slightly (example 0.010)

If it never detects contact:
Decrease CONTACT_CURRENT_RISE_A (example 0.25)
Increase Z_DESCENT_STEP_M slightly (example 0.0012)

This is not true impedance, but it behaves like a soft compliant Z axis at the tip without adding force sensors.