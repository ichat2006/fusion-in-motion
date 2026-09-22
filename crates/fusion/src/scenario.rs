use std::{collections::BTreeSet, fs, path::Path};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedScenario {
    pub root_seed: u64,
    #[serde(default = "one")]
    pub motion_speed_factor: f64,
    pub world: WorldConfig,
    pub trajectory: Vec<MotionSegment>,
    #[serde(default)]
    pub imu: ImuConfig,
    #[serde(default)]
    pub gps: GpsConfig,
    #[serde(default)]
    pub camera: CameraConfig,
    #[serde(default)]
    pub lidar: LidarConfig,
    #[serde(default)]
    pub ego_estimator: EgoEstimatorConfig,
    #[serde(default)]
    pub object_tracker: ObjectTrackerConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

impl ResolvedScenario {
    pub fn duration_s(&self) -> f64 {
        self.trajectory
            .iter()
            .map(|segment| segment.duration_s)
            .sum()
    }

    pub fn effective_duration_s(&self) -> f64 {
        self.duration_s() / self.motion_speed_factor
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldConfig {
    pub objects: Vec<ObjectConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectConfig {
    pub id: String,
    pub initial_position_m: Vec2Config,
    pub velocity_world_mps: Vec2Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotionSegment {
    pub id: String,
    pub duration_s: f64,
    pub longitudinal_acceleration_mps2: f64,
    pub yaw_rate_radps: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vec2Config {
    pub x: f64,
    pub y: f64,
}

impl Vec2Config {
    fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImuConfig {
    pub enabled: bool,
    pub rate_hz: f64,
    pub latency_ns: i64,
    pub clock_offset_ns: i64,
    pub gyro_white_noise_density_radps_sqrt_hz: f64,
    pub accel_white_noise_density_mps2_sqrt_hz: f64,
    pub gyro_bias_radps: f64,
    pub accel_bias_mps2: f64,
    pub gyro_bias_random_walk_radps_sqrt_s: f64,
    pub accel_bias_random_walk_mps2_sqrt_s: f64,
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rate_hz: 100.0,
            latency_ns: 0,
            clock_offset_ns: 0,
            gyro_white_noise_density_radps_sqrt_hz: 0.0008,
            accel_white_noise_density_mps2_sqrt_hz: 0.012,
            gyro_bias_radps: 0.0,
            accel_bias_mps2: 0.0,
            gyro_bias_random_walk_radps_sqrt_s: 0.0,
            accel_bias_random_walk_mps2_sqrt_s: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpsConfig {
    pub enabled: bool,
    pub rate_hz: f64,
    pub latency_ns: i64,
    pub horizontal_position_stddev_m: f64,
    pub outlier_probability: f64,
    pub outlier_stddev_m: f64,
    pub dropout_start_s: Option<f64>,
    pub dropout_end_s: Option<f64>,
}

impl Default for GpsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rate_hz: 2.0,
            latency_ns: 0,
            horizontal_position_stddev_m: 0.25,
            outlier_probability: 0.0,
            outlier_stddev_m: 10.0,
            dropout_start_s: None,
            dropout_end_s: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraConfig {
    pub enabled: bool,
    pub rate_hz: f64,
    pub latency_ns: i64,
    pub horizontal_fov_rad: f64,
    pub max_range_m: f64,
    pub bearing_noise_stddev_rad: f64,
    pub detection_probability: f64,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rate_hz: 10.0,
            latency_ns: 0,
            horizontal_fov_rad: 2.8,
            max_range_m: 25.0,
            bearing_noise_stddev_rad: 0.006,
            detection_probability: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LidarConfig {
    pub enabled: bool,
    pub rate_hz: f64,
    pub latency_ns: i64,
    pub horizontal_fov_rad: f64,
    pub max_range_m: f64,
    pub scan_duration_ns: i64,
    pub range_noise_stddev_m: f64,
    pub bearing_noise_stddev_rad: f64,
    pub detection_probability: f64,
}

impl Default for LidarConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rate_hz: 5.0,
            latency_ns: 0,
            horizontal_fov_rad: std::f64::consts::TAU,
            max_range_m: 25.0,
            scan_duration_ns: 0,
            range_noise_stddev_m: 0.025,
            bearing_noise_stddev_rad: 0.002,
            detection_probability: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EgoEstimatorAlgorithm {
    #[default]
    Basic,
    ImuBias,
    GtsamEkfPlanar,
}

impl EgoEstimatorAlgorithm {
    pub const fn estimates_imu_bias(self) -> bool {
        matches!(self, Self::ImuBias | Self::GtsamEkfPlanar)
    }

    pub const fn state_dimension(self) -> usize {
        match self {
            Self::Basic => 4,
            Self::ImuBias | Self::GtsamEkfPlanar => 6,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Basic => "basic",
            Self::ImuBias => "imu_bias",
            Self::GtsamEkfPlanar => "gtsam_ekf_planar",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EgoEstimatorConfig {
    pub algorithm: EgoEstimatorAlgorithm,
    pub timing_compensation: bool,
    pub history_duration_ns: i64,
    pub gps_gate_sigma: f64,
    pub initial_position_stddev_m: f64,
    pub initial_yaw_stddev_rad: f64,
    pub initial_speed_stddev_mps: f64,
    pub initial_gyro_bias_stddev_radps: f64,
    pub initial_accel_bias_stddev_mps2: f64,
}

impl Default for EgoEstimatorConfig {
    fn default() -> Self {
        Self {
            algorithm: EgoEstimatorAlgorithm::Basic,
            timing_compensation: false,
            history_duration_ns: 1_000_000_000,
            gps_gate_sigma: 1.0e9,
            initial_position_stddev_m: 0.5,
            initial_yaw_stddev_rad: 0.1,
            initial_speed_stddev_mps: 0.5,
            initial_gyro_bias_stddev_radps: 0.1,
            initial_accel_bias_stddev_mps2: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObjectTrackerConfig {
    pub timing_compensation: bool,
    pub history_duration_ns: i64,
    pub acceleration_noise_stddev_mps2: f64,
    pub gate_sigma: f64,
    pub confirmation_hits: usize,
    pub max_time_without_update_s: f64,
}

impl Default for ObjectTrackerConfig {
    fn default() -> Self {
        Self {
            timing_compensation: false,
            history_duration_ns: 1_000_000_000,
            acceleration_noise_stddev_mps2: 0.5,
            gate_sigma: 4.0,
            confirmation_hits: 2,
            max_time_without_update_s: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetricsConfig {
    pub max_truth_match_gap_ns: i64,
    pub track_truth_match_max_distance_m: f64,
    pub ego_divergence_position_error_m: f64,
    pub track_divergence_position_error_m: f64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            max_truth_match_gap_ns: 10_000_000,
            track_truth_match_max_distance_m: 5.0,
            ego_divergence_position_error_m: 5.0,
            track_divergence_position_error_m: 5.0,
        }
    }
}

pub fn load_and_resolve(path: &Path) -> Result<ResolvedScenario> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("failed to read scenario {}", path.display()))?;
    let scenario: ResolvedScenario = serde_yaml_ng::from_str(&source)
        .with_context(|| format!("invalid scenario {}", path.display()))?;
    validate(&scenario)?;
    Ok(scenario)
}

pub fn validate(scenario: &ResolvedScenario) -> Result<()> {
    ensure!(
        scenario.motion_speed_factor.is_finite() && scenario.motion_speed_factor > 0.0,
        "motion_speed_factor must be positive"
    );
    ensure!(
        scenario.imu.enabled,
        "the vehicle estimator requires the IMU"
    );
    for (name, enabled, rate, latency) in [
        (
            "imu",
            scenario.imu.enabled,
            scenario.imu.rate_hz,
            scenario.imu.latency_ns,
        ),
        (
            "gps",
            scenario.gps.enabled,
            scenario.gps.rate_hz,
            scenario.gps.latency_ns,
        ),
        (
            "camera",
            scenario.camera.enabled,
            scenario.camera.rate_hz,
            scenario.camera.latency_ns,
        ),
        (
            "lidar",
            scenario.lidar.enabled,
            scenario.lidar.rate_hz,
            scenario.lidar.latency_ns,
        ),
    ] {
        if enabled {
            validate_rate(name, rate)?;
        }
        ensure!(latency >= 0, "{name} latency must be nonnegative");
    }
    ensure!(
        scenario.imu.clock_offset_ns >= 0,
        "IMU clock offset must be nonnegative"
    );
    for (name, value) in [
        (
            "gyro noise",
            scenario.imu.gyro_white_noise_density_radps_sqrt_hz,
        ),
        (
            "accelerometer noise",
            scenario.imu.accel_white_noise_density_mps2_sqrt_hz,
        ),
        (
            "gyro bias walk",
            scenario.imu.gyro_bias_random_walk_radps_sqrt_s,
        ),
        (
            "accelerometer bias walk",
            scenario.imu.accel_bias_random_walk_mps2_sqrt_s,
        ),
        ("GPS noise", scenario.gps.horizontal_position_stddev_m),
        ("GPS outlier size", scenario.gps.outlier_stddev_m),
        (
            "camera bearing noise",
            scenario.camera.bearing_noise_stddev_rad,
        ),
        ("lidar range noise", scenario.lidar.range_noise_stddev_m),
        (
            "lidar bearing noise",
            scenario.lidar.bearing_noise_stddev_rad,
        ),
    ] {
        ensure!(
            value.is_finite() && value >= 0.0,
            "{name} must be finite and nonnegative"
        );
    }
    validate_detection_sensor("camera", &scenario.camera)?;
    ensure!(
        (0.0..=1.0).contains(&scenario.gps.outlier_probability),
        "GPS outlier probability must be in [0, 1]"
    );
    validate_lidar(&scenario.lidar)?;
    validate_world(&scenario.world)?;
    validate_trajectory(&scenario.trajectory)?;
    validate_estimators(scenario)?;
    Ok(())
}

fn validate_detection_sensor(name: &str, config: &CameraConfig) -> Result<()> {
    ensure!(
        config.horizontal_fov_rad > 0.0 && config.horizontal_fov_rad <= std::f64::consts::TAU,
        "{name} field of view must be in (0, 2pi]"
    );
    ensure!(
        config.max_range_m.is_finite() && config.max_range_m > 0.0,
        "{name} range must be positive"
    );
    ensure!(
        (0.0..=1.0).contains(&config.detection_probability),
        "{name} detection probability must be in [0, 1]"
    );
    Ok(())
}

fn validate_lidar(config: &LidarConfig) -> Result<()> {
    ensure!(
        config.horizontal_fov_rad > 0.0 && config.horizontal_fov_rad <= std::f64::consts::TAU,
        "lidar field of view must be in (0, 2pi]"
    );
    ensure!(
        config.max_range_m.is_finite() && config.max_range_m > 0.0,
        "lidar range must be positive"
    );
    ensure!(
        config.scan_duration_ns >= 0,
        "lidar scan duration must be nonnegative"
    );
    ensure!(
        (0.0..=1.0).contains(&config.detection_probability),
        "lidar detection probability must be in [0, 1]"
    );
    Ok(())
}

fn validate_world(world: &WorldConfig) -> Result<()> {
    ensure!(
        !world.objects.is_empty(),
        "world must contain at least one object"
    );
    let mut ids = BTreeSet::new();
    for object in &world.objects {
        ensure!(
            !object.id.is_empty() && ids.insert(&object.id),
            "object IDs must be nonempty and unique"
        );
        ensure!(
            object.initial_position_m.finite() && object.velocity_world_mps.finite(),
            "object {} state must be finite",
            object.id
        );
    }
    Ok(())
}

fn validate_trajectory(trajectory: &[MotionSegment]) -> Result<()> {
    ensure!(
        !trajectory.is_empty(),
        "trajectory must contain at least one segment"
    );
    let mut ids = BTreeSet::new();
    let mut speed = 0.0;
    for segment in trajectory {
        ensure!(
            !segment.id.is_empty() && ids.insert(&segment.id),
            "trajectory IDs must be nonempty and unique"
        );
        ensure!(
            segment.duration_s.is_finite() && segment.duration_s > 0.0,
            "segment {} duration must be positive",
            segment.id
        );
        ensure!(
            segment.longitudinal_acceleration_mps2.is_finite()
                && segment.yaw_rate_radps.is_finite(),
            "segment {} motion must be finite",
            segment.id
        );
        speed += segment.longitudinal_acceleration_mps2 * segment.duration_s;
        ensure!(
            speed >= -1.0e-9,
            "segment {} produces a negative forward speed",
            segment.id
        );
    }
    Ok(())
}

fn validate_estimators(scenario: &ResolvedScenario) -> Result<()> {
    ensure!(
        scenario.ego_estimator.history_duration_ns >= 0
            && scenario.object_tracker.history_duration_ns >= 0,
        "history durations must be nonnegative"
    );
    for (name, value) in [
        ("GPS gate", scenario.ego_estimator.gps_gate_sigma),
        (
            "initial position uncertainty",
            scenario.ego_estimator.initial_position_stddev_m,
        ),
        (
            "initial yaw uncertainty",
            scenario.ego_estimator.initial_yaw_stddev_rad,
        ),
        (
            "initial speed uncertainty",
            scenario.ego_estimator.initial_speed_stddev_mps,
        ),
        (
            "initial gyro bias uncertainty",
            scenario.ego_estimator.initial_gyro_bias_stddev_radps,
        ),
        (
            "initial accelerometer bias uncertainty",
            scenario.ego_estimator.initial_accel_bias_stddev_mps2,
        ),
        (
            "tracker acceleration noise",
            scenario.object_tracker.acceleration_noise_stddev_mps2,
        ),
        ("tracker gate", scenario.object_tracker.gate_sigma),
        (
            "tracker maximum time without an update",
            scenario.object_tracker.max_time_without_update_s,
        ),
        (
            "track truth match distance",
            scenario.metrics.track_truth_match_max_distance_m,
        ),
    ] {
        ensure!(
            value.is_finite() && value > 0.0,
            "{name} must be positive and finite"
        );
    }
    ensure!(
        scenario.metrics.max_truth_match_gap_ns >= 0,
        "truth match gap must be nonnegative"
    );
    ensure!(
        scenario.object_tracker.confirmation_hits > 0,
        "tracker confirmation hits must be positive"
    );
    ensure!(
        scenario.metrics.ego_divergence_position_error_m > 0.0
            && scenario.metrics.track_divergence_position_error_m > 0.0,
        "error thresholds must be positive"
    );
    Ok(())
}

fn validate_rate(name: &str, value: f64) -> Result<()> {
    ensure!(
        value.is_finite() && value > 0.0,
        "{name} rate must be positive"
    );
    let period_ns = 1.0e9 / value;
    ensure!(
        (period_ns - period_ns.round()).abs() < 1.0e-6,
        "{name} rate must map to an integer nanosecond period"
    );
    Ok(())
}

pub fn canonical_yaml(scenario: &ResolvedScenario) -> Result<String> {
    Ok(serde_yaml_ng::to_string(scenario)?)
}

fn one() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_fields_are_rejected() {
        let yaml = "root_seed: 1\nunknown: true\n";
        assert!(serde_yaml_ng::from_str::<ResolvedScenario>(yaml).is_err());
    }
}
