
import os

import matplotlib
matplotlib.use('Agg')  # headless - no display needed to save PNGs from inside a node
import matplotlib.pyplot as plt
import numpy as np


def _to_arrays(samples: list):
    t  = np.array([s['t'] for s in samples])
    x  = np.array([s['x'] for s in samples])
    y  = np.array([s['y'] for s in samples])
    vx = np.array([s.get('vx', 0.0) for s in samples])
    vy = np.array([s.get('vy', 0.0) for s in samples])
    vz = np.array([s.get('vz', 0.0) for s in samples])
    order = np.argsort(t)
    return t[order], x[order], y[order], vx[order], vy[order], vz[order]


def _median_dt(t: np.ndarray):
    if len(t) < 2:
        return None
    dt = np.median(np.diff(t))
    return float(dt) if dt > 0 else None


def _spectrum(t: np.ndarray, signal: np.ndarray):
    dt = _median_dt(t)
    if dt is None or len(signal) < 4:
        return None, None
    centered = signal - np.mean(signal)
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    mag = np.abs(np.fft.rfft(centered))
    peak = mag.max()
    if peak > 0:
        mag = mag / peak
    return freqs, mag


def plot_position(gt_samples, est_samples, output_dir, robot_name):
    t_gt, x_gt, y_gt, *_ = _to_arrays(gt_samples)
    t_est, x_est, y_est, *_ = _to_arrays(est_samples)
    t0 = min(t_gt[0], t_est[0])
    t_gt_rel, t_est_rel = t_gt - t0, t_est - t0

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t_gt_rel, x_gt, color='teal', label='hook_ground_truth_base_flat')
    axes[0].plot(t_est_rel, x_est, color='crimson', marker='.', label='hook_state (estimate)')
    axes[0].set_ylabel('position x [m]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_gt_rel, y_gt, color='teal', label='hook_ground_truth_base_flat')
    axes[1].plot(t_est_rel, y_est, color='crimson', marker='.', label='hook_state (estimate)')
    axes[1].set_ylabel('position y [m]')
    axes[1].set_xlabel('time [s]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Hook position in frame {robot_name}/base_flat_link — estimate vs ground truth')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'position_comparison.png'), dpi=150)
    plt.close(fig)


def plot_diagnostics(gt_samples, est_samples, output_dir, robot_name, L=None):
    t_gt, x_gt, y_gt, *_ = _to_arrays(gt_samples)
    t_est, x_est, y_est, *_ = _to_arrays(est_samples)
    t0 = min(t_gt[0], t_est[0])
    t_gt_rel, t_est_rel = t_gt - t0, t_est - t0

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]
    ax.plot(x_gt, y_gt, color='teal', label='hook_ground_truth_base_flat')
    ax.plot(x_est, y_est, color='crimson', marker='.', label='hook_state (estimate)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('XY trajectory (same frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    ax = axs[0, 1]
    ax.plot(t_gt_rel, x_gt - np.mean(x_gt), color='teal', label='ground truth x (de-meaned)')
    ax.plot(t_est_rel, y_est - np.mean(y_est), color='crimson', marker='.', label='estimate y (de-meaned)')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('[m]')
    ax.set_title('The two moving axes, superimposed')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axs[1, 0]
    f_gt_x, m_gt_x = _spectrum(t_gt, x_gt)
    f_gt_y, m_gt_y = _spectrum(t_gt, y_gt)
    f_est_x, m_est_x = _spectrum(t_est, x_est)
    f_est_y, m_est_y = _spectrum(t_est, y_est)

    if f_gt_x is not None:
        ax.semilogy(f_gt_x, np.clip(m_gt_x, 1e-4, None), color='teal', label='hook_ground_truth_base_flat px')
        ax.semilogy(f_gt_y, np.clip(m_gt_y, 1e-4, None), color='teal', linestyle='--', label='hook_ground_truth_base_flat py')
    if f_est_x is not None:
        ax.semilogy(f_est_x, np.clip(m_est_x, 1e-4, None), color='crimson', label='hook_state px')
        ax.semilogy(f_est_y, np.clip(m_est_y, 1e-4, None), color='crimson', linestyle='--', label='hook_state py')

    if L is not None and L > 0:
        f_n = np.sqrt(9.81 / L) / (2 * np.pi)
        ax.axvline(f_n, color='gray', linestyle=':', label=f'pendulum fn for L={L:.2f} m')

    dt_est = _median_dt(t_est)
    if dt_est:
        nyq_est = 0.5 / dt_est
        ax.axvline(nyq_est, color='lightgray', linestyle=':', label=f'estimate Nyquist ({nyq_est:.2f} Hz)')

    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('normalised |FFT|')
    ax.set_title('Spectra of the position signals')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axs[1, 1]
    med_gt = _median_dt(t_gt)
    med_est = _median_dt(t_est)
    gt_label = f'hook_ground_truth_base_flat  (med {med_gt:.3f}s / {1/med_gt:.1f} Hz)' if med_gt else 'hook_ground_truth_base_flat'
    est_label = f'hook_state (estimate)  (med {med_est:.3f}s / {1/med_est:.1f} Hz)' if med_est else 'hook_state (estimate)'
    if len(t_gt_rel) > 1:
        ax.plot(t_gt_rel[1:], np.diff(t_gt_rel), color='teal', label=gt_label)
    if len(t_est_rel) > 1:
        ax.plot(t_est_rel[1:], np.diff(t_est_rel), color='crimson', label=est_label)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Δt between messages [s]')
    ax.set_title('Publication interval')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'diagnostics.png'), dpi=150)
    plt.close(fig)


def plot_raw_measurement(gt_samples, raw_samples, output_dir, robot_name):
    
    t_gt, x_gt, y_gt, *_ = _to_arrays(gt_samples)
    t_raw, x_raw, y_raw, *_ = _to_arrays(raw_samples)
    t0 = min(t_gt[0], t_raw[0])
    t_gt_rel, t_raw_rel = t_gt - t0, t_raw - t0

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t_gt_rel, x_gt, color='teal', label='hook_ground_truth_base_flat')
    axes[0].plot(t_raw_rel, x_raw, color='darkorange', marker='.', linestyle='none',
                 label='raw measurement (pre-fusion)')
    axes[0].set_ylabel('position x [m]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_gt_rel, y_gt, color='teal', label='hook_ground_truth_base_flat')
    axes[1].plot(t_raw_rel, y_raw, color='darkorange', marker='.', linestyle='none',
                 label='raw measurement (pre-fusion)')
    axes[1].set_ylabel('position y [m]')
    axes[1].set_xlabel('time [s]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Raw hook measurement vs ground truth in {robot_name}/base_flat_link '
                 f'(axis-mapping check)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'raw_measurement_vs_gt.png'), dpi=150)
    plt.close(fig)


def plot_velocity(gt_samples, est_samples, output_dir, robot_name):
    t_gt, _, _, vx_gt, vy_gt, vz_gt = _to_arrays(gt_samples)
    t_est, _, _, vx_est, vy_est, vz_est = _to_arrays(est_samples)
    t0 = min(t_gt[0], t_est[0])
    t_gt_rel, t_est_rel = t_gt - t0, t_est - t0

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t_gt_rel, vx_gt, color='teal', label='hook_ground_truth_base_flat')
    axes[0].plot(t_est_rel, vx_est, color='crimson', marker='.', label='hook_state (estimate)')
    axes[0].set_ylabel('linear vel x [m/s]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_gt_rel, vy_gt, color='teal', label='hook_ground_truth_base_flat')
    axes[1].plot(t_est_rel, vy_est, color='crimson', marker='.', label='hook_state (estimate)')
    axes[1].set_ylabel('linear vel y [m/s]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    est_vz_label = 'hook_state (estimate)'
    if np.allclose(vz_est, 0.0):
        est_vz_label += '  [identically zero - not populated]'
    axes[2].plot(t_gt_rel, vz_gt, color='teal', label='hook_ground_truth_base_flat')
    axes[2].plot(t_est_rel, vz_est, color='crimson', linestyle='--', label=est_vz_label)
    axes[2].set_ylabel('linear vel z [m/s]')
    axes[2].set_xlabel('time since bag start (header.stamp) [s]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Hook linear velocity — estimate vs ground truth')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'velocity_comparison.png'), dpi=150)
    plt.close(fig)


def save_all_plots(gt_samples: list, est_samples: list, output_dir: str,
                    robot_name: str, L: "float|None" = None,
                    raw_samples: "list|None" = None) -> "tuple[bool, str]":
    """Saves position_comparison.png, diagnostics.png, velocity_comparison.png
    into output_dir. If raw_samples (the pre-fusion per-detection measurement) is
    provided and non-empty, also saves raw_measurement_vs_gt.png. Returns
    (success, message)."""
    if len(gt_samples) < 2 or len(est_samples) < 2:
        return False, (f"Not enough data to plot (ground truth samples: {len(gt_samples)}, "
                        f"estimate samples: {len(est_samples)}) - need at least 2 of each")

    os.makedirs(output_dir, exist_ok=True)

    plot_position(gt_samples, est_samples, output_dir, robot_name)
    plot_diagnostics(gt_samples, est_samples, output_dir, robot_name, L=L)
    plot_velocity(gt_samples, est_samples, output_dir, robot_name)

    saved = "position_comparison.png, diagnostics.png, velocity_comparison.png"
    if raw_samples is not None and len(raw_samples) >= 2:
        plot_raw_measurement(gt_samples, raw_samples, output_dir, robot_name)
        saved += ", raw_measurement_vs_gt.png"
    else:
        saved += (f" (no raw_measurement_vs_gt.png - only {len(raw_samples) if raw_samples else 0} "
                  f"raw samples; is hook_raw_measurement being published?)")

    return True, f"Saved {saved} to {output_dir}"
