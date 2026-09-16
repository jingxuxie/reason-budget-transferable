"""Budget-transferable selection with exact verification and fixed continuations."""
from .core import (Segment, compress_feasible, lazy_selector, regret, robust_gaps,
                   validate_profiles, empirical_cdf, hazard_cdf)
from .racing import CappedObservation, DeadlineRacer, run_race, radius
