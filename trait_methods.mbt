///| Explicitly expose derived trait methods as regular methods.

///|
/// Newer MoonBit toolchains no longer implicitly promote these methods.
pub extend TraceAccumulator with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend CandidateSet with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Checkpoint with Eq::{equal, not_equal}

///|
pub extend Checkpoint with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Config with Eq::{equal, not_equal}

///|
pub extend Config with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Version with Eq::{equal, not_equal}

///|
pub extend Version with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Step with Eq::{equal, not_equal}

///|
pub extend Step with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Sampler with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TraceSummary with Eq::{equal, not_equal}

///|
pub extend TraceSummary with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend StopReason with Eq::{equal, not_equal}

///|
pub extend StopReason with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Generation with Eq::{equal, not_equal}

///|
pub extend Generation with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend StreamGeneration with Eq::{equal, not_equal}

///|
pub extend StreamGeneration with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PreparedRow with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend Preview with Eq::{equal, not_equal}

///|
pub extend Preview with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SamplingError with Eq::{equal, not_equal}

///|
pub extend SamplingError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ParkMiller with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TraceSpread with Eq::{equal, not_equal}

///|
pub extend TraceSpread with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SamplerState with Eq::{equal, not_equal}

///|
pub extend SamplerState with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SurpriseWindow with @moonbitlang/core/debug.Debug::{to_repr}
