"""Plots of what the controller did, built from published topics only.

Previously the mission plots were assembled inside
`alars_move_to_damped_action_server` from its own internal state, which let them
show things nothing publishes - the feedforward/trim split, the shaped plan, the
STABILIZING->MOVING boundary. That coupling is gone: the action server now only
flies, and everything here is reconstructed from hook_swing_state, cmd_vel and
smarc/odom, so anything that can see those topics can produce these plots.

Sample dicts are produced by SwayPlotter; each carries a 't' in seconds (the
header stamp) plus whatever fields its topic has.
"""

import os

import matplotlib
matplotlib.use('Agg')  # headless - no display needed to save PNGs from inside a node
import matplotlib.pyplot as plt
import numpy as np


def _col(samples, key):
    return np.array([s.get(key, np.nan) for s in samples], dtype=float)


def _t_rel(samples, t0):
    return _col(samples, 't') - t0


def _start_time(*sample_lists):
    """Common time origin, so every plot shares one x axis."""
    starts = [s[0]['t'] for s in sample_lists if s]
    return min(starts) if starts else 0.0


def plot_swing(swing_samples, output_dir, robot_name, t0, theta_tol=None):
    """The payload itself. This is the plot that says whether the sway damping
    worked: the swing should decay while the drone holds station and stay small
    through a move, rather than being excited by it."""
    t = _t_rel(swing_samples, t0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, _col(swing_samples, 'theta_x'), color='crimson', label='theta_x')
    axes[0].plot(t, _col(swing_samples, 'theta_y'), color='teal', label='theta_y')
    if theta_tol:
        axes[0].axhspan(-theta_tol, theta_tol, color='0.85', zorder=0,
                        label=f'settle tolerance (+/-{theta_tol:.3f})')
    axes[0].set_ylabel('swing angle [rad]')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, _col(swing_samples, 'omega_x'), color='crimson', label='omega_x')
    axes[1].plot(t, _col(swing_samples, 'omega_y'), color='teal', label='omega_y')
    axes[1].set_ylabel('swing rate [rad/s]')
    axes[1].set_xlabel('time [s]')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'{robot_name} payload swing (hook_swing_state)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'swing.png'), dpi=150)
    plt.close(fig)


def plot_command(cmd_samples, odom_samples, output_dir, robot_name, t0):
    """What was commanded, and whether the drone followed it.

    The per-axis panels show the command alone: cmd_vel is expressed in
    base_flat_link while smarc/odom's twist is in the odom frame (dji_captain
    fills it from the ground velocity), so overlaying their x/y would compare
    two different frames. The SPEED panel is the honest comparison - magnitude
    does not depend on the frame."""
    t_cmd = _t_rel(cmd_samples, t0)
    ux, uy = _col(cmd_samples, 'vx'), _col(cmd_samples, 'vy')

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t_cmd, ux, color='crimson', label='commanded vx (base_flat)')
    axes[0].set_ylabel('velocity x [m/s]')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_cmd, uy, color='teal', label='commanded vy (base_flat)')
    axes[1].set_ylabel('velocity y [m/s]')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_cmd, np.hypot(ux, uy), color='black', label='commanded speed')
    if odom_samples:
        t_od = _t_rel(odom_samples, t0)
        axes[2].plot(t_od, np.hypot(_col(odom_samples, 'vx'), _col(odom_samples, 'vy')),
                     color='0.5', label='achieved speed (smarc/odom)')
    axes[2].set_ylabel('horizontal speed [m/s]')
    axes[2].set_xlabel('time [s]')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'{robot_name} velocity command (cmd_vel) vs achieved speed')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'command.png'), dpi=150)
    plt.close(fig)


def plot_trajectory(odom_samples, output_dir, robot_name, t0):
    """Where the drone actually went. There is no plan to compare against here -
    the shaped reference exists only inside the action server - so this answers
    "what path did it fly", not "did it track the plan"."""
    t = _t_rel(odom_samples, t0)
    px, py = _col(odom_samples, 'x'), _col(odom_samples, 'y')

    fig = plt.figure(figsize=(13, 6))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(px, py, color='crimson', label='drone')
    if len(px):
        ax.plot(px[0], py[0], 'o', color='green', markersize=7, label='start')
        ax.plot(px[-1], py[-1], 'o', color='black', markersize=7, label='end')
    ax.set_xlabel('x [m]  (odom)')
    ax.set_ylabel('y [m]  (odom)')
    ax.set_title('trajectory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, px, color='crimson', label='drone x')
    ax.set_ylabel('x [m]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, py, color='teal', label='drone y')
    ax.set_ylabel('y [m]')
    ax.set_xlabel('time [s]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{robot_name} drone trajectory (smarc/odom)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'trajectory.png'), dpi=150)
    plt.close(fig)


def summarise(swing_samples, cmd_samples) -> str:
    """One-line verdict, so the log says what happened without opening a PNG."""
    parts = []
    if swing_samples:
        th = np.abs(np.concatenate([_col(swing_samples, 'theta_x'),
                                    _col(swing_samples, 'theta_y')]))
        th = th[np.isfinite(th)]
        parts.append(f'{len(swing_samples)} swing samples')
        if th.size:
            parts.append(f'peak |theta| {th.max():.4f} rad')
    if cmd_samples:
        speed = np.hypot(_col(cmd_samples, 'vx'), _col(cmd_samples, 'vy'))
        speed = speed[np.isfinite(speed)]
        if speed.size:
            parts.append(f'peak commanded speed {speed.max():.3f} m/s')
    return ', '.join(parts) if parts else 'nothing recorded'


def save_control_plots(swing_samples: list, cmd_samples: list, odom_samples: list,
                       output_dir: str, robot_name: str,
                       theta_tol: "float|None" = None) -> "tuple[bool, str]":
    """Saves whichever of swing.png, command.png and trajectory.png the recorded
    topics support. A topic that was never published costs its own plot and
    nothing else."""
    saved = []
    t0 = _start_time(swing_samples, cmd_samples, odom_samples)

    if len(swing_samples) >= 2:
        os.makedirs(output_dir, exist_ok=True)
        plot_swing(swing_samples, output_dir, robot_name, t0, theta_tol=theta_tol)
        saved.append('swing.png')
    if len(cmd_samples) >= 2:
        os.makedirs(output_dir, exist_ok=True)
        plot_command(cmd_samples, odom_samples, output_dir, robot_name, t0)
        saved.append('command.png')
    if len(odom_samples) >= 2:
        os.makedirs(output_dir, exist_ok=True)
        plot_trajectory(odom_samples, output_dir, robot_name, t0)
        saved.append('trajectory.png')

    if not saved:
        return False, (f'No control plots (hook_swing_state: {len(swing_samples)}, '
                       f'cmd_vel: {len(cmd_samples)}, smarc/odom: {len(odom_samples)} '
                       f'samples) - need at least 2 samples of a topic to plot it')

    return True, (f'Saved {", ".join(saved)} to {output_dir} - '
                  f'{summarise(swing_samples, cmd_samples)}')
