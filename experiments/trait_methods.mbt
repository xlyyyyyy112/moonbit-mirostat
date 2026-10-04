///|
/// Compatibility declarations for explicit derived trait methods.
pub extend Baseline with Eq::{equal, not_equal}

///|
pub extend Baseline with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend BaselineStep with Eq::{equal, not_equal}

///|
pub extend BaselineStep with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Comparison with Eq::{equal, not_equal}

///|
pub extend Comparison with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Scenario with Eq::{equal, not_equal}

///|
pub extend Scenario with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SweepPoint with Eq::{equal, not_equal}

///|
pub extend SweepPoint with @moonbitlang/core/debug.Debug::{to_repr}
