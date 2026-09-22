use fusion_schema::messages::{GpsFix, Vec2};

use crate::{
    bundle::MeasurementRecord, random::DeterministicRandom, scenario::ResolvedScenario,
    truth::Trajectory,
};

use super::{PendingMeasurement, for_each_sample, measurement_time};

const DELIVERY_PRIORITY: u8 = 15;

pub(super) fn generate(
    scenario: &ResolvedScenario,
    trajectory: &Trajectory,
    random: &DeterministicRandom,
    output: &mut Vec<PendingMeasurement>,
) {
    let config = &scenario.gps;
    for_each_sample(
        scenario.effective_duration_s(),
        config.rate_hz,
        |time_ns, sequence| {
            let time_s = time_ns as f64 / 1e9;
            if let (Some(start), Some(end)) = (config.dropout_start_s, config.dropout_end_s)
                && time_s >= start
                && time_s < end
            {
                return;
            }
            let platform = trajectory.sample_ns(time_ns);
            let outlier_scale =
                if random.uniform("gps", sequence, "outlier", 0) < config.outlier_probability {
                    config.outlier_stddev_m
                } else {
                    0.0
                };
            let position = Vec2 {
                x: platform.x_world_m
                    + config.horizontal_position_stddev_m
                        * random.normal("gps", sequence, "position_noise", 0)
                    + outlier_scale * random.normal("gps", sequence, "outlier_noise", 0),
                y: platform.y_world_m
                    + config.horizontal_position_stddev_m
                        * random.normal("gps", sequence, "position_noise", 1)
                    + outlier_scale * random.normal("gps", sequence, "outlier_noise", 1),
            };
            let arrival_ns = time_ns + config.latency_ns;
            output.push(PendingMeasurement {
                arrival_ns,
                priority: DELIVERY_PRIORITY,
                stable_event_id: format!("gps:{sequence:010}"),
                measurement: MeasurementRecord::Gps(GpsFix {
                    time: Some(measurement_time(time_ns, arrival_ns)),
                    position_world_m: Some(position),
                    horizontal_position_variance_m2: config.horizontal_position_stddev_m.powi(2),
                }),
                imu_bias_truth: None,
            });
        },
    );
}
