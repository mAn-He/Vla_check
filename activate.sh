cd ~/Vla_check
source .venvs/libero/bin/activate
export MUJOCO_GL=osmesa
echo "ready: $(python -c 'import sys; print(sys.executable)')"
