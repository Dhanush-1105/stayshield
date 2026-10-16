"""Small helpers shared by all scripts."""
import matplotlib.pyplot as plt

import config


def banner(title):
    """Print a clear section heading."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def finish_plot(name):
    """Save the current figure to outputs/ and show it if allowed."""
    plt.tight_layout()
    path = config.OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=130)
    print(f"[plot saved] {path.relative_to(config.PROJECT_ROOT)}")
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()


def require_preprocessed():
    """
    The clustering scripts need data/hotel_bookings_preprocessed.csv.
    If it does not exist yet, build it by running preprocessing.py.
    """
    if not config.PREPROCESSED_CSV.exists():
        import runpy
        print("Preprocessed file not found - running preprocessing.py first...")
        runpy.run_path(str(config.PROJECT_ROOT / "preprocessing.py"))
