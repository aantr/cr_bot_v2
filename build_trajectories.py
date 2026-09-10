"""Convenience entry point: python build_trajectories.py battle.mp4 --result win."""
import sys

from offline_rl.build_trajectories import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled; no completed trajectory was written.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
