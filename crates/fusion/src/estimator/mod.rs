mod basic;
#[cfg(feature = "gtsam")]
mod gtsam;
mod imu_bias;

use anyhow::{Context, Result, ensure};
use fusion_schema::messages::{EgoStateEstimate, GpsFix, ImuSample, MeasurementTime};
use serde::{Deserialize, Serialize};

use crate::scenario::{EgoEstimatorAlgorithm, EgoEstimatorConfig, ImuConfig};

#[cfg(feature = "gtsam")]
use self::gtsam::GtsamEkfPlanarEstimator;
use self::{basic::BasicEkf, imu_bias::ImuBiasEkf};

#[derive(Debug, Clone, Copy, PartialEq)]
enum UpdateResult {
    Applied { normalized_residual: f64 },
    Rejected { normalized_residual: f64 },
    Invalid,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EgoMeasurement {
    Imu(ImuSample),
    Gps(GpsFix),
}

impl EgoMeasurement {
    pub fn time(&self) -> &MeasurementTime {
        match self {
            Self::Imu(value) => value.time.as_ref(),
            Self::Gps(value) => value.time.as_ref(),
        }
        .expect("generated ego measurements have time")
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpsDiagnostics {
    pub attempted_fixes: usize,
    pub accepted_fixes: usize,
    pub rejected_fixes: usize,
    pub invalid_fixes: usize,
    pub maximum_normalized_residual: f64,
}

impl GpsDiagnostics {
    fn record(&mut self, result: UpdateResult) {
        self.attempted_fixes += 1;
        match result {
            UpdateResult::Applied {
                normalized_residual,
            } => {
                self.accepted_fixes += 1;
                self.maximum_normalized_residual =
                    self.maximum_normalized_residual.max(normalized_residual);
            }
            UpdateResult::Rejected {
                normalized_residual,
            } => {
                self.rejected_fixes += 1;
                self.maximum_normalized_residual =
                    self.maximum_normalized_residual.max(normalized_residual);
            }
            UpdateResult::Invalid => self.invalid_fixes += 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorFreshness {
    /// Ages at the last arrival in the complete ego input stream, not wall time.
    /// None means this sensor never delivered a packet.
    pub final_receipt_age_ns: Option<i64>,
    pub final_measurement_age_ns: Option<i64>,
    /// Longest interval between deliveries, including trailing silence.
    /// Time before the first packet is excluded because its start is unknown.
    pub maximum_receipt_gap_ns: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingDiagnostics {
    pub timing_compensation: bool,
    pub history_duration_ns: i64,
    pub received_measurements: usize,
    pub delayed_measurements: usize,
    pub replayed_measurements: usize,
    pub discarded_measurements: usize,
    pub revised_estimates: usize,
    pub maximum_delivery_age_ns: i64,
    pub freshness_observed_at_ns: Option<i64>,
    pub gps_freshness: SensorFreshness,
    pub imu_freshness: SensorFreshness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatorAssumptions {
    pub algorithm: EgoEstimatorAlgorithm,
    pub backend_version: Option<String>,
    pub state_order: Vec<String>,
    pub initial_covariance_diagonal: Vec<f64>,
    pub imu_process_noise: ImuProcessNoise,
    pub gps_gate_sigma: f64,
}

#[derive(Debug)]
pub struct EstimatorRun {
    pub estimates: Vec<EgoStateEstimate>,
    pub timing: TimingDiagnostics,
    pub gps_diagnostics: GpsDiagnostics,
    pub assumptions: EstimatorAssumptions,
}

pub fn run(
    config: &EgoEstimatorConfig,
    imu: &ImuConfig,
    measurements: &[EgoMeasurement],
) -> Result<EstimatorRun> {
    validate_delivery_order(measurements)?;
    let settings = EstimatorSettings::resolve(config, imu);
    let filter = ActiveEstimator::new(config.algorithm, &settings)
        .with_context(|| format!("failed to construct {} estimator", config.algorithm.name()))?;
    let assumptions = filter.assumptions(config.algorithm, &settings);
    if config.timing_compensation {
        run_at_measurement_time(config, measurements, assumptions, settings, filter)
    } else {
        run_at_arrival(config, measurements, assumptions, settings, filter)
    }
}

fn run_at_arrival(
    config: &EgoEstimatorConfig,
    measurements: &[EgoMeasurement],
    assumptions: EstimatorAssumptions,
    settings: EstimatorSettings,
    mut estimator: ActiveEstimator,
) -> Result<EstimatorRun> {
    let mut gps_diagnostics = GpsDiagnostics::default();
    let mut estimates = Vec::new();
    let mut latest_imu_stamp_ns = None;
    let mut delayed = 0;
    for measurement in measurements {
        let time = measurement.time();
        if latest_imu_stamp_ns.is_some_and(|latest| time.measurement_time_ns < latest) {
            delayed += 1;
        }
        match measurement {
            EgoMeasurement::Imu(imu) => {
                estimator
                    .propagate(imu, &settings.imu_process_noise)
                    .with_context(|| estimator_error(config.algorithm, "IMU", time))?;
                latest_imu_stamp_ns = Some(time.measurement_time_ns);
                let estimate = estimator
                    .estimate(time.measurement_time_ns, time.arrival_time_ns)
                    .with_context(|| estimator_error(config.algorithm, "IMU", time))?;
                validate_estimate(config.algorithm, &estimate)
                    .with_context(|| estimator_error(config.algorithm, "IMU", time))?;
                estimates.push(estimate);
            }
            EgoMeasurement::Gps(fix) => {
                let result = estimator
                    .update_gps(fix, settings.gps_gate_sigma)
                    .with_context(|| estimator_error(config.algorithm, "GPS", time))?;
                gps_diagnostics.record(result);
            }
        }
    }
    Ok(EstimatorRun {
        estimates,
        timing: timing(config, measurements, delayed, 0, 0, 0),
        gps_diagnostics,
        assumptions,
    })
}

fn run_at_measurement_time(
    config: &EgoEstimatorConfig,
    measurements: &[EgoMeasurement],
    assumptions: EstimatorAssumptions,
    settings: EstimatorSettings,
    mut estimator: ActiveEstimator,
) -> Result<EstimatorRun> {
    let mut accepted = Vec::new();
    let mut latest_imu_stamp_ns = None;
    let mut delayed = 0;
    let mut replayed = 0;
    let mut discarded = 0;
    for measurement in measurements {
        let time = measurement.time();
        let age = latest_imu_stamp_ns
            .map(|latest: i64| latest.saturating_sub(time.measurement_time_ns))
            .unwrap_or(0);
        if age > 0 {
            delayed += 1;
        }
        if matches!(measurement, EgoMeasurement::Gps(_)) && age > config.history_duration_ns {
            discarded += 1;
        } else {
            if age > 0 {
                replayed += 1;
            }
            accepted.push(measurement);
        }
        if matches!(measurement, EgoMeasurement::Imu(_)) {
            latest_imu_stamp_ns = Some(
                latest_imu_stamp_ns.map_or(time.measurement_time_ns, |time: i64| {
                    time.max(measurement.time().measurement_time_ns)
                }),
            );
        }
    }
    accepted.sort_by_key(|measurement| {
        (
            measurement.time().measurement_time_ns,
            measurement_priority(measurement),
            measurement.time().arrival_time_ns,
        )
    });

    let mut gps_diagnostics = GpsDiagnostics::default();
    let mut estimates = Vec::new();
    let mut revised = 0;
    let mut index = 0;
    while index < accepted.len() {
        let stamp = accepted[index].time().measurement_time_ns;
        let end = index
            + accepted[index..]
                .partition_point(|measurement| measurement.time().measurement_time_ns == stamp);
        let mut emission = None;
        for measurement in &accepted[index..end] {
            let time = measurement.time();
            match measurement {
                EgoMeasurement::Imu(imu) => {
                    estimator
                        .propagate(imu, &settings.imu_process_noise)
                        .with_context(|| estimator_error(config.algorithm, "IMU", time))?;
                    emission = Some(time);
                }
                EgoMeasurement::Gps(fix) => {
                    let result = estimator
                        .update_gps(fix, settings.gps_gate_sigma)
                        .with_context(|| estimator_error(config.algorithm, "GPS", time))?;
                    gps_diagnostics.record(result);
                }
            }
        }
        if let Some(imu_time) = emission {
            let mut final_emission = imu_time.arrival_time_ns;
            let mut was_revised = false;
            for measurement in &accepted {
                let time = measurement.time();
                if matches!(measurement, EgoMeasurement::Gps(_))
                    && time.measurement_time_ns <= stamp
                    && time.arrival_time_ns > imu_time.arrival_time_ns
                {
                    final_emission = final_emission.max(time.arrival_time_ns);
                    was_revised = true;
                }
            }
            revised += usize::from(was_revised);
            let estimate = estimator
                .estimate(stamp, final_emission)
                .with_context(|| estimator_error(config.algorithm, "IMU", imu_time))?;
            validate_estimate(config.algorithm, &estimate)
                .with_context(|| estimator_error(config.algorithm, "IMU", imu_time))?;
            estimates.push(estimate);
        }
        index = end;
    }
    Ok(EstimatorRun {
        estimates,
        timing: timing(config, measurements, delayed, replayed, discarded, revised),
        gps_diagnostics,
        assumptions,
    })
}

fn timing(
    config: &EgoEstimatorConfig,
    measurements: &[EgoMeasurement],
    delayed_measurements: usize,
    replayed_measurements: usize,
    discarded_measurements: usize,
    revised_estimates: usize,
) -> TimingDiagnostics {
    TimingDiagnostics {
        timing_compensation: config.timing_compensation,
        history_duration_ns: config.history_duration_ns,
        received_measurements: measurements.len(),
        delayed_measurements,
        replayed_measurements,
        discarded_measurements,
        revised_estimates,
        maximum_delivery_age_ns: measurements
            .iter()
            .map(|measurement| {
                let time = measurement.time();
                time.arrival_time_ns
                    .saturating_sub(time.measurement_time_ns)
            })
            .max()
            .unwrap_or(0),
        freshness_observed_at_ns: measurements
            .last()
            .map(|value| value.time().arrival_time_ns),
        gps_freshness: sensor_freshness(measurements, |value| {
            matches!(value, EgoMeasurement::Gps(_))
        }),
        imu_freshness: sensor_freshness(measurements, |value| {
            matches!(value, EgoMeasurement::Imu(_))
        }),
    }
}

// Inspect original deliveries, never reordered/replayed filter updates. A rejected
// or history-discarded GPS fix still represents a packet received from the sensor.
fn sensor_freshness(
    measurements: &[EgoMeasurement],
    is_sensor: impl Fn(&EgoMeasurement) -> bool,
) -> SensorFreshness {
    let mut last_arrival: Option<i64> = None;
    let mut latest_measurement: Option<i64> = None;
    let mut maximum_gap = 0;
    for measurement in measurements.iter().filter(|value| is_sensor(value)) {
        let time = measurement.time();
        if let Some(previous) = last_arrival {
            maximum_gap = maximum_gap.max(time.arrival_time_ns.saturating_sub(previous));
        }
        last_arrival = Some(time.arrival_time_ns);
        latest_measurement = Some(
            latest_measurement.map_or(time.measurement_time_ns, |previous| {
                previous.max(time.measurement_time_ns)
            }),
        );
    }
    let observed_at = measurements
        .last()
        .map(|value| value.time().arrival_time_ns);
    let receipt_age = observed_at
        .zip(last_arrival)
        .map(|(now, last)| now.saturating_sub(last));
    SensorFreshness {
        final_receipt_age_ns: receipt_age,
        final_measurement_age_ns: observed_at
            .zip(latest_measurement)
            .map(|(now, latest)| now.saturating_sub(latest)),
        maximum_receipt_gap_ns: receipt_age.map(|age| maximum_gap.max(age)),
    }
}

fn measurement_priority(measurement: &EgoMeasurement) -> u8 {
    match measurement {
        EgoMeasurement::Imu(_) => 10,
        EgoMeasurement::Gps(_) => 20,
    }
}

fn validate_delivery_order(measurements: &[EgoMeasurement]) -> Result<()> {
    let mut previous = None;
    for measurement in measurements {
        let delivery = measurement.time().arrival_time_ns;
        ensure!(
            delivery >= measurement.time().measurement_time_ns,
            "ego measurement arrives before its measurement time"
        );
        if let Some(previous) = previous {
            ensure!(
                delivery >= previous,
                "ego measurements are not in arrival order"
            );
        }
        previous = Some(delivery);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImuProcessNoise {
    pub gyro_white_noise_density_radps_sqrt_hz: f64,
    pub accel_white_noise_density_mps2_sqrt_hz: f64,
    pub gyro_bias_random_walk_radps_sqrt_s: f64,
    pub accel_bias_random_walk_mps2_sqrt_s: f64,
}

impl ImuProcessNoise {
    fn for_algorithm(config: &ImuConfig, algorithm: EgoEstimatorAlgorithm) -> Self {
        let estimates_bias = algorithm.estimates_imu_bias();
        Self {
            gyro_white_noise_density_radps_sqrt_hz: config.gyro_white_noise_density_radps_sqrt_hz,
            accel_white_noise_density_mps2_sqrt_hz: config.accel_white_noise_density_mps2_sqrt_hz,
            gyro_bias_random_walk_radps_sqrt_s: if estimates_bias {
                config.gyro_bias_random_walk_radps_sqrt_s
            } else {
                0.0
            },
            accel_bias_random_walk_mps2_sqrt_s: if estimates_bias {
                config.accel_bias_random_walk_mps2_sqrt_s
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EstimatorSettings {
    initial_position_variance_m2: f64,
    initial_yaw_variance_rad2: f64,
    initial_speed_variance_m2ps2: f64,
    initial_gyro_bias_variance_rad2ps2: f64,
    initial_accel_bias_variance_m2ps4: f64,
    imu_process_noise: ImuProcessNoise,
    gps_gate_sigma: f64,
}

impl EstimatorSettings {
    fn resolve(config: &EgoEstimatorConfig, imu: &ImuConfig) -> Self {
        Self {
            initial_position_variance_m2: config.initial_position_stddev_m.powi(2),
            initial_yaw_variance_rad2: config.initial_yaw_stddev_rad.powi(2),
            initial_speed_variance_m2ps2: config.initial_speed_stddev_mps.powi(2),
            initial_gyro_bias_variance_rad2ps2: config.initial_gyro_bias_stddev_radps.powi(2),
            initial_accel_bias_variance_m2ps4: config.initial_accel_bias_stddev_mps2.powi(2),
            imu_process_noise: ImuProcessNoise::for_algorithm(imu, config.algorithm),
            gps_gate_sigma: config.gps_gate_sigma,
        }
    }
}

enum ActiveEstimator {
    Basic(BasicEkf),
    ImuBias(ImuBiasEkf),
    #[cfg(feature = "gtsam")]
    GtsamEkfPlanar(GtsamEkfPlanarEstimator),
}

impl ActiveEstimator {
    fn new(algorithm: EgoEstimatorAlgorithm, settings: &EstimatorSettings) -> Result<Self> {
        Ok(match algorithm {
            EgoEstimatorAlgorithm::Basic => Self::Basic(BasicEkf::new(settings)),
            EgoEstimatorAlgorithm::ImuBias => Self::ImuBias(ImuBiasEkf::new(settings)),
            #[cfg(feature = "gtsam")]
            EgoEstimatorAlgorithm::GtsamEkfPlanar => {
                Self::GtsamEkfPlanar(GtsamEkfPlanarEstimator::new(settings)?)
            }
            #[cfg(not(feature = "gtsam"))]
            EgoEstimatorAlgorithm::GtsamEkfPlanar => {
                anyhow::bail!("gtsam_ekf_planar requires a build with `--features gtsam`")
            }
        })
    }

    fn assumptions(
        &self,
        algorithm: EgoEstimatorAlgorithm,
        settings: &EstimatorSettings,
    ) -> EstimatorAssumptions {
        let (state_names, initial_covariance_diagonal): (&[&str], Vec<f64>) = match self {
            Self::Basic(filter) => (&basic::STATE_NAMES, filter.covariance_diagonal()),
            Self::ImuBias(filter) => (&imu_bias::STATE_NAMES, filter.covariance_diagonal()),
            #[cfg(feature = "gtsam")]
            Self::GtsamEkfPlanar(_) => (
                &imu_bias::STATE_NAMES,
                vec![
                    settings.initial_position_variance_m2,
                    settings.initial_position_variance_m2,
                    settings.initial_yaw_variance_rad2,
                    settings.initial_speed_variance_m2ps2,
                    settings.initial_gyro_bias_variance_rad2ps2,
                    settings.initial_accel_bias_variance_m2ps4,
                ],
            ),
        };
        EstimatorAssumptions {
            algorithm,
            backend_version: match self {
                #[cfg(feature = "gtsam")]
                Self::GtsamEkfPlanar(_) => {
                    Some(format!("GTSAM {}", GtsamEkfPlanarEstimator::version()))
                }
                _ => None,
            },
            state_order: state_names.iter().map(|name| (*name).to_owned()).collect(),
            initial_covariance_diagonal,
            imu_process_noise: settings.imu_process_noise,
            gps_gate_sigma: settings.gps_gate_sigma,
        }
    }

    fn propagate(&mut self, imu: &ImuSample, noise: &ImuProcessNoise) -> Result<()> {
        match self {
            Self::Basic(filter) => filter.propagate(imu, noise),
            Self::ImuBias(filter) => filter.propagate(imu, noise),
            #[cfg(feature = "gtsam")]
            Self::GtsamEkfPlanar(filter) => filter.propagate(imu),
        }
    }

    fn update_gps(&mut self, fix: &GpsFix, gps_gate_sigma: f64) -> Result<UpdateResult> {
        match self {
            Self::Basic(filter) => filter.update_gps(fix, gps_gate_sigma),
            Self::ImuBias(filter) => filter.update_gps(fix, gps_gate_sigma),
            #[cfg(feature = "gtsam")]
            Self::GtsamEkfPlanar(filter) => filter.update_gps(fix),
        }
    }

    fn estimate(&self, estimate_time_ns: i64, available_time_ns: i64) -> Result<EgoStateEstimate> {
        match self {
            Self::Basic(filter) => Ok(filter.estimate(estimate_time_ns, available_time_ns)),
            Self::ImuBias(filter) => Ok(filter.estimate(estimate_time_ns, available_time_ns)),
            #[cfg(feature = "gtsam")]
            Self::GtsamEkfPlanar(filter) => filter.estimate(estimate_time_ns, available_time_ns),
        }
    }
}

fn validate_estimate(algorithm: EgoEstimatorAlgorithm, estimate: &EgoStateEstimate) -> Result<()> {
    let pose = estimate
        .pose_world
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("estimate has no pose"))?;
    let position = pose
        .position
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("estimate pose has no position"))?;
    ensure!(
        [
            position.x,
            position.y,
            pose.yaw_rad,
            estimate.forward_speed_mps,
        ]
        .into_iter()
        .all(f64::is_finite),
        "estimate state contains a non-finite value"
    );

    let state_dimension = algorithm.state_dimension();
    ensure!(
        estimate.state_covariance.len() == state_dimension * state_dimension,
        "estimate covariance has {} values; expected {}",
        estimate.state_covariance.len(),
        state_dimension * state_dimension
    );
    ensure!(
        estimate
            .state_covariance
            .iter()
            .all(|value| value.is_finite()),
        "estimate covariance contains a non-finite value"
    );
    for index in 0..state_dimension {
        ensure!(
            estimate.state_covariance[index * state_dimension + index] >= 0.0,
            "estimate covariance diagonal {index} is negative"
        );
    }

    if algorithm.estimates_imu_bias() {
        ensure!(
            estimate.gyro_bias_z_radps.is_some_and(f64::is_finite),
            "estimate has no finite gyroscope bias"
        );
        ensure!(
            estimate.accel_bias_x_mps2.is_some_and(f64::is_finite),
            "estimate has no finite accelerometer bias"
        );
    }
    Ok(())
}

fn estimator_error(
    algorithm: EgoEstimatorAlgorithm,
    sensor: &str,
    time: &MeasurementTime,
) -> String {
    format!(
        "{} estimator failed on {sensor} measured at {} ns and received at {} ns",
        algorithm.name(),
        time.measurement_time_ns,
        time.arrival_time_ns
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timed_measurement(gps: bool, stamp: i64, arrival: i64) -> EgoMeasurement {
        let time = Some(MeasurementTime {
            measurement_time_ns: stamp,
            arrival_time_ns: arrival,
        });
        if gps {
            EgoMeasurement::Gps(GpsFix {
                time,
                ..Default::default()
            })
        } else {
            EgoMeasurement::Imu(ImuSample {
                time,
                ..Default::default()
            })
        }
    }

    // Existing estimator tests check filter math and replay, not the distinction
    // between packet receipt, measurement freshness, and trailing sensor silence.
    #[test]
    fn freshness_preserves_newest_stamp_and_counts_trailing_silence() {
        let measurements = [
            timed_measurement(true, 0, 0),
            timed_measurement(true, 8, 10),
            timed_measurement(true, 4, 11), // late old packet must not replace stamp 8
            timed_measurement(false, 25, 25),
        ];
        let diagnostic = timing(&EgoEstimatorConfig::default(), &measurements, 0, 0, 0, 0);
        assert_eq!(diagnostic.freshness_observed_at_ns, Some(25));
        assert_eq!(diagnostic.gps_freshness.final_receipt_age_ns, Some(14));
        assert_eq!(diagnostic.gps_freshness.final_measurement_age_ns, Some(17));
        assert_eq!(diagnostic.gps_freshness.maximum_receipt_gap_ns, Some(14));
        assert_eq!(diagnostic.imu_freshness.final_measurement_age_ns, Some(0));
    }

    #[test]
    fn freshness_distinguishes_recovery_from_never_received() {
        let measurements = [
            timed_measurement(true, 4_500_000_000, 4_500_000_000),
            timed_measurement(true, 10_000_000_000, 10_250_000_000),
        ];
        let diagnostic = timing(&EgoEstimatorConfig::default(), &measurements, 0, 0, 0, 0);
        assert_eq!(diagnostic.gps_freshness.final_receipt_age_ns, Some(0));
        assert_eq!(
            diagnostic.gps_freshness.final_measurement_age_ns,
            Some(250_000_000)
        );
        assert_eq!(
            diagnostic.gps_freshness.maximum_receipt_gap_ns,
            Some(5_750_000_000)
        );
        assert_eq!(diagnostic.imu_freshness.final_receipt_age_ns, None);
        assert_eq!(diagnostic.imu_freshness.final_measurement_age_ns, None);
        assert_eq!(diagnostic.imu_freshness.maximum_receipt_gap_ns, None);
        let empty = timing(&EgoEstimatorConfig::default(), &[], 0, 0, 0, 0);
        assert_eq!(empty.freshness_observed_at_ns, None);
        assert_eq!(empty.gps_freshness.final_receipt_age_ns, None);
    }

    #[test]
    fn measurement_cannot_arrive_before_it_was_taken() {
        assert!(validate_delivery_order(&[timed_measurement(true, 2, 1)]).is_err());
    }

    #[test]
    fn rejected_and_history_discarded_gps_still_count_as_received() -> Result<()> {
        let mut gps = timed_measurement(true, 4, 11);
        if let EgoMeasurement::Gps(fix) = &mut gps {
            fix.position_world_m = Some(fusion_schema::messages::Vec2 { x: 1e6, y: 1e6 });
            fix.horizontal_position_variance_m2 = 1.0;
        }
        let measurements = [
            timed_measurement(false, 1, 1),
            timed_measurement(false, 10, 10),
            gps,
        ];
        for algorithm in [EgoEstimatorAlgorithm::Basic, EgoEstimatorAlgorithm::ImuBias] {
            for timing_compensation in [false, true] {
                let config = EgoEstimatorConfig {
                    algorithm,
                    timing_compensation,
                    history_duration_ns: 1,
                    gps_gate_sigma: 3.0,
                    ..Default::default()
                };
                let result = run(&config, &ImuConfig::default(), &measurements)?;
                assert_eq!(result.timing.gps_freshness.final_receipt_age_ns, Some(0));
                assert_eq!(
                    result.timing.gps_freshness.final_measurement_age_ns,
                    Some(7)
                );
                assert_eq!(result.timing.imu_freshness.final_receipt_age_ns, Some(1));
                if timing_compensation {
                    assert_eq!(result.timing.discarded_measurements, 1);
                    assert_eq!(result.gps_diagnostics.attempted_fixes, 0);
                } else {
                    assert_eq!(result.gps_diagnostics.rejected_fixes, 1);
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "gtsam")]
    use fusion_schema::messages::Vec2;

    fn estimate(algorithm: EgoEstimatorAlgorithm) -> EgoStateEstimate {
        let state_dimension = algorithm.state_dimension();
        EgoStateEstimate {
            pose_world: Some(crate::math::pose2(0.0, 0.0, 0.0)),
            state_covariance: vec![0.0; state_dimension * state_dimension],
            gyro_bias_z_radps: algorithm.estimates_imu_bias().then_some(0.0),
            accel_bias_x_mps2: algorithm.estimates_imu_bias().then_some(0.0),
            ..Default::default()
        }
    }

    #[test]
    fn estimator_output_must_match_its_declared_state() {
        let basic = estimate(EgoEstimatorAlgorithm::Basic);
        assert!(validate_estimate(EgoEstimatorAlgorithm::Basic, &basic).is_ok());

        let mut non_finite_state = basic.clone();
        non_finite_state.forward_speed_mps = f64::NAN;
        assert!(validate_estimate(EgoEstimatorAlgorithm::Basic, &non_finite_state).is_err());

        let mut wrong_covariance_size = basic.clone();
        wrong_covariance_size.state_covariance.pop();
        assert!(validate_estimate(EgoEstimatorAlgorithm::Basic, &wrong_covariance_size).is_err());

        let mut negative_variance = basic;
        negative_variance.state_covariance[0] = -1.0;
        assert!(validate_estimate(EgoEstimatorAlgorithm::Basic, &negative_variance).is_err());

        let mut missing_bias = estimate(EgoEstimatorAlgorithm::ImuBias);
        missing_bias.gyro_bias_z_radps = None;
        assert!(validate_estimate(EgoEstimatorAlgorithm::ImuBias, &missing_bias).is_err());
    }

    #[cfg(feature = "gtsam")]
    #[test]
    fn gtsam_ekf_matches_the_rust_bias_ekf_after_every_input() -> Result<()> {
        let config = EgoEstimatorConfig {
            algorithm: EgoEstimatorAlgorithm::GtsamEkfPlanar,
            gps_gate_sigma: 3.0,
            ..Default::default()
        };
        let imu_config = ImuConfig::default();
        let settings = EstimatorSettings::resolve(&config, &imu_config);
        let mut rust = ImuBiasEkf::new(&settings);
        let mut gtsam = GtsamEkfPlanarEstimator::new(&settings)?;

        for (index, measurement) in [
            imu_sample(0, 0.02, 0.4),
            imu_sample(100_000_000, 0.03, 0.4),
            imu_sample(200_000_000, 0.20, 0.1),
        ]
        .iter()
        .enumerate()
        {
            rust.propagate(measurement, &settings.imu_process_noise)?;
            gtsam.propagate(measurement)?;
            assert_estimates_match(
                index,
                "IMU",
                measurement.time.as_ref().unwrap(),
                &rust,
                &gtsam,
            )?;
        }

        for (index, fix) in [
            gps_fix(200_000_000, 0.03, -0.02, 0.09),
            gps_fix(200_000_000, 100.0, 100.0, 0.09),
        ]
        .iter()
        .enumerate()
        {
            let rust_result = rust.update_gps(fix, settings.gps_gate_sigma)?;
            let gtsam_result = gtsam.update_gps(fix)?;
            assert_update_matches(index, rust_result, gtsam_result);
            assert_estimates_match(index, "GPS", fix.time.as_ref().unwrap(), &rust, &gtsam)?;
        }
        Ok(())
    }

    #[cfg(feature = "gtsam")]
    fn imu_sample(time_ns: i64, yaw_rate_radps: f64, acceleration_mps2: f64) -> ImuSample {
        ImuSample {
            time: Some(MeasurementTime {
                measurement_time_ns: time_ns,
                arrival_time_ns: time_ns,
            }),
            yaw_rate_radps,
            forward_acceleration_mps2: acceleration_mps2,
        }
    }

    #[cfg(feature = "gtsam")]
    fn gps_fix(time_ns: i64, x_m: f64, y_m: f64, variance_m2: f64) -> GpsFix {
        GpsFix {
            time: Some(MeasurementTime {
                measurement_time_ns: time_ns,
                arrival_time_ns: time_ns,
            }),
            position_world_m: Some(Vec2 { x: x_m, y: y_m }),
            horizontal_position_variance_m2: variance_m2,
        }
    }

    #[cfg(feature = "gtsam")]
    fn assert_update_matches(index: usize, rust: UpdateResult, gtsam: UpdateResult) {
        match (rust, gtsam) {
            (
                UpdateResult::Applied {
                    normalized_residual: rust,
                },
                UpdateResult::Applied {
                    normalized_residual: gtsam,
                },
            )
            | (
                UpdateResult::Rejected {
                    normalized_residual: rust,
                },
                UpdateResult::Rejected {
                    normalized_residual: gtsam,
                },
            ) => assert!(
                (rust - gtsam).abs() < 1.0e-8,
                "GPS input {index} normalized residual differs: Rust {rust}, GTSAM {gtsam}"
            ),
            (UpdateResult::Invalid, UpdateResult::Invalid) => {}
            (rust, gtsam) => {
                panic!("GPS input {index} decision differs: Rust {rust:?}, GTSAM {gtsam:?}")
            }
        }
    }

    #[cfg(feature = "gtsam")]
    fn assert_estimates_match(
        index: usize,
        sensor: &str,
        time: &MeasurementTime,
        rust: &ImuBiasEkf,
        gtsam: &GtsamEkfPlanarEstimator,
    ) -> Result<()> {
        let rust = rust.estimate(time.measurement_time_ns, time.arrival_time_ns);
        let gtsam = gtsam.estimate(time.measurement_time_ns, time.arrival_time_ns)?;
        let state = |estimate: &EgoStateEstimate| {
            let pose = estimate.pose_world.as_ref().unwrap();
            let position = pose.position.as_ref().unwrap();
            [
                position.x,
                position.y,
                pose.yaw_rad,
                estimate.forward_speed_mps,
                estimate.gyro_bias_z_radps.unwrap(),
                estimate.accel_bias_x_mps2.unwrap(),
            ]
        };
        for (coordinate, (rust, gtsam)) in state(&rust).into_iter().zip(state(&gtsam)).enumerate() {
            assert!(
                (rust - gtsam).abs() < 1.0e-8,
                "{sensor} input {index}, measured at {} ns: state[{coordinate}] differs: Rust {rust}, GTSAM {gtsam}",
                time.measurement_time_ns
            );
        }
        for (coordinate, (rust, gtsam)) in rust
            .state_covariance
            .iter()
            .zip(&gtsam.state_covariance)
            .enumerate()
        {
            assert!(
                (rust - gtsam).abs() < 1.0e-8,
                "{sensor} input {index}, measured at {} ns: covariance[{coordinate}] differs: Rust {rust}, GTSAM {gtsam}",
                time.measurement_time_ns
            );
        }
        Ok(())
    }
}
