
import os

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np


def _col(samples, key):
    return np.array([s.get(key, np.nan) for s in samples], dtype=float)


def _phase_boundary(samples):
    """Time at which STABILIZING handed over to MOVING, or None."""
    for prev, cur in zip(samples, samples[1:]):
        if prev.get('phase') == 'STABILIZING' and cur.get('phase') == 'MOVING':
            return float(cur['t'])
    return None


def _mark_phase(ax, t_switch):
    if t_switch is not None:
        ax.axvline(t_switch, color='0.4', linestyle='--', linewidth=1.2)
        ax.annotate('mission start', xy=(t_switch, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords='offset points',
                    fontsize=8, color='0.3')


def plot_swing(samples, output_dir, robot_name, theta_tol=None):
    """The payload itself. This is the plot that says whether the sway damping
    worked: swing should decay during STABILIZING and stay small through the
    move, rather than being excited by it."""
    t = _col(samples, 't')
    t_switch = _phase_boundary(samples)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, _col(samples, 'theta_x'), color='crimson', label='theta_x')
    axes[0].plot(t, _col(samples, 'theta_y'), color='teal', label='theta_y')
    if theta_tol:
        axes[0].axhspan(-theta_tol, theta_tol, color='0.85', zorder=0,
                        label=f'settle tolerance (+/-{theta_tol:.3f})')
    axes[0].set_ylabel('swing angle [rad]')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    _mark_phase(axes[0], t_switch)

    axes[1].plot(t, _col(samples, 'omega_x'), color='crimson', label='omega_x')
    axes[1].plot(t, _col(samples, 'omega_y'), color='teal', label='omega_y')
    axes[1].set_ylabel('swing rate [rad/s]')
    axes[1].set_xlabel('time since goal [s]')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    _mark_phase(axes[1], t_switch)

    fig.suptitle(f'{robot_name} move_to_dumped - payload swing')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'mission_swing.png'), dpi=150)
    plt.close(fig)


def plot_tracking(samples, output_dir, robot_name):
    """Did it go where the plan said, and when."""
    t = _col(samples, 't')
    t_switch = _phase_boundary(samples)
    px, py = _col(samples, 'p_x'), _col(samples, 'p_y')
    rx, ry = _col(samples, 'p_ref_x'), _col(samples, 'p_ref_y')

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(rx, ry, color='0.5', linestyle='--', label='shaped plan')
    ax.plot(px, py, color='crimson', label='drone')
    if len(px):
        ax.plot(px[0], py[0], 'o', color='green', markersize=7, label='start')
        ax.plot(px[-1], py[-1], 'o', color='black', markersize=7, label='end')
    ax.set_xlabel('x [m]  (map)')
    ax.set_ylabel('y [m]  (map)')
    ax.set_title('trajectory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, rx, color='0.5', linestyle='--', label='plan x')
    ax.plot(t, px, color='crimson', label='drone x')
    ax.set_ylabel('x [m]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _mark_phase(ax, t_switch)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, ry, color='0.5', linestyle='--', label='plan y')
    ax.plot(t, py, color='teal', label='drone y')
    ax.set_ylabel('y [m]')
    ax.set_xlabel('time since goal [s]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _mark_phase(ax, t_switch)

    fig.suptitle(f'{robot_name} move_to_dumped - plan tracking')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'mission_tracking.png'), dpi=150)
    plt.close(fig)


def plot_command(samples, output_dir, robot_name):
    """Feedforward vs feedback. The ZVD plan is supposed to do nearly all the
    work, so a trim that is persistently large means the plan and the plant
    disagree - wrong L/xi, or a real disturbance."""
    t = _col(samples, 't')
    t_switch = _phase_boundary(samples)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, axis, colour in ((axes[0], 'x', 'crimson'), (axes[1], 'y', 'teal')):
        ax.plot(t, _col(samples, f'v_ff_{axis}'), color='0.5', linestyle='--',
                label=f'feedforward (ZVD) {axis}')
        ax.plot(t, _col(samples, f'trim_{axis}'), color=colour, linewidth=1.0,
                label=f'LQG trim {axis}')
        ax.plot(t, _col(samples, f'u_{axis}'), color='black', linewidth=1.4,
                label=f'published {axis}')
        ax.set_ylabel(f'velocity {axis} [m/s]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _mark_phase(ax, t_switch)
    axes[1].set_xlabel('time since goal [s]')

    fig.suptitle(f'{robot_name} move_to_dumped - command breakdown (base_flat_link)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'mission_command.png'), dpi=150)
    plt.close(fig)


def summarise(samples, theta_tol=None) -> str:
    """One-line verdict, so the log says what happened without opening a PNG."""
    if not samples:
        return 'no samples recorded'
    th = np.abs(np.concatenate([_col(samples, 'theta_x'), _col(samples, 'theta_y')]))
    trim = np.abs(np.concatenate([_col(samples, 'trim_x'), _col(samples, 'trim_y')]))
    th = th[np.isfinite(th)]
    trim = trim[np.isfinite(trim)]
    moving = [s for s in samples if s.get('phase') == 'MOVING']
    parts = [f'{len(samples)} samples over {samples[-1]["t"]:.1f}s']
    if th.size:
        parts.append(f'peak |theta| {th.max():.4f} rad')
    if moving:
        thm = np.abs(np.concatenate([_col(moving, 'theta_x'), _col(moving, 'theta_y')]))
        thm = thm[np.isfinite(thm)]
        if thm.size:
            parts.append(f'peak |theta| while moving {thm.max():.4f} rad')
    if trim.size:
        parts.append(f'peak |trim| {trim.max():.3f} m/s, mean {trim.mean():.3f}')
    return ', '.join(parts)


def save_mission_plots(samples: list, output_dir: str, robot_name: str,
                       theta_tol: "float|None" = None) -> "tuple[bool, str]":
    """Saves mission_swing.png, mission_tracking.png, mission_command.png into
    output_dir. Returns (success, message)."""
    if len(samples) < 2:
        return False, f'Not enough samples to plot the mission ({len(samples)})'

    os.makedirs(output_dir, exist_ok=True)
    plot_swing(samples, output_dir, robot_name, theta_tol=theta_tol)
    plot_tracking(samples, output_dir, robot_name)
    plot_command(samples, output_dir, robot_name)

    return True, (f'Saved mission_swing.png, mission_tracking.png, mission_command.png '
                  f'to {output_dir} - {summarise(samples, theta_tol)}')
