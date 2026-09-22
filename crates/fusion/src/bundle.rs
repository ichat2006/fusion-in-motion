use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::BufWriter,
    path::Path,
};

use anyhow::{Context, Result, bail};
use fusion_schema::{
    FILE_DESCRIPTOR_SET,
    messages::{
        CameraFrame, EgoStateEstimate, EgoTruthState, GpsFix, ImuBiasTruth, ImuSample, LidarScan,
        MeasurementTime, ObjectTrackFrame, ObjectTruthState,
    },
};
use mcap::{Writer, records::MessageHeader};
use prost::Message;

use crate::{
    estimator::{EstimatorAssumptions, GpsDiagnostics, TimingDiagnostics},
    eval::RunMetrics,
    scenario::{ResolvedScenario, canonical_yaml},
    tracker::{TrackerDiagnostics, TrackerHistory},
};

#[derive(Debug, Clone)]
pub enum MeasurementRecord {
    Imu(ImuSample),
    Gps(GpsFix),
    Camera(CameraFrame),
    Lidar(LidarScan),
}

impl MeasurementRecord {
    pub fn time(&self) -> &MeasurementTime {
        match self {
            Self::Imu(value) => value.time.as_ref(),
            Self::Gps(value) => value.time.as_ref(),
            Self::Camera(value) => value.time.as_ref(),
            Self::Lidar(value) => value.time.as_ref(),
        }
        .expect("generated measurements have time")
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedRun {
    pub measurements: Vec<MeasurementRecord>,
    pub ego_truth_states: Vec<EgoTruthState>,
    pub object_truth_states: Vec<ObjectTruthState>,
    pub imu_bias_truth: Vec<ImuBiasTruth>,
}

pub fn prepare(output: &Path, scenario: &ResolvedScenario) -> Result<()> {
    if output.exists() {
        bail!("output directory {} already exists", output.display());
    }
    fs::create_dir_all(output.join("estimates"))?;
    fs::create_dir_all(output.join("tracks"))?;
    fs::create_dir_all(output.join("reports/baseline"))?;
    fs::write(
        output.join("scenario.resolved.yaml"),
        canonical_yaml(scenario)?,
    )?;
    Ok(())
}

pub fn write_generated(output: &Path, generated: &GeneratedRun) -> Result<()> {
    write_measurements(&output.join("measurements.mcap"), &generated.measurements)?;
    write_truth(
        &output.join("truth.mcap"),
        &generated.ego_truth_states,
        &generated.object_truth_states,
        &generated.imu_bias_truth,
    )
}

pub fn write_ego_estimates(output: &Path, estimates: &[EgoStateEstimate]) -> Result<()> {
    write_ego_estimates_file(
        &output.join("estimates/ego-baseline.mcap"),
        "ego-baseline",
        estimates,
    )
}

pub fn write_ego_estimates_file(
    path: &Path,
    name: &str,
    estimates: &[EgoStateEstimate],
) -> Result<()> {
    let mut writer = new_writer(path)?;
    let schema = writer.add_schema("fusion.EgoStateEstimate", "protobuf", FILE_DESCRIPTOR_SET)?;
    let channel = writer.add_channel(
        schema,
        &format!("/estimate/ego/{name}"),
        "protobuf",
        &BTreeMap::new(),
    )?;
    for (sequence, estimate) in estimates.iter().enumerate() {
        write_message(
            &mut writer,
            channel,
            sequence as u32,
            estimate.available_time_ns,
            estimate.estimate_time_ns,
            &estimate.encode_to_vec(),
        )?;
    }
    writer.finish()?;
    Ok(())
}

pub fn write_tracks(output: &Path, name: &str, frames: &[ObjectTrackFrame]) -> Result<()> {
    write_tracks_file(
        &output.join("tracks").join(format!("{name}.mcap")),
        name,
        frames,
    )
}

pub fn write_tracks_file(path: &Path, name: &str, frames: &[ObjectTrackFrame]) -> Result<()> {
    let mut writer = new_writer(path)?;
    let schema = writer.add_schema("fusion.ObjectTrackFrame", "protobuf", FILE_DESCRIPTOR_SET)?;
    let channel = writer.add_channel(
        schema,
        &format!("/track/object/{name}"),
        "protobuf",
        &BTreeMap::new(),
    )?;
    for (sequence, frame) in frames.iter().enumerate() {
        write_message(
            &mut writer,
            channel,
            sequence as u32,
            frame.available_time_ns,
            frame.estimate_time_ns,
            &frame.encode_to_vec(),
        )?;
    }
    writer.finish()?;
    Ok(())
}

pub fn read_measurements(path: &Path) -> Result<Vec<MeasurementRecord>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut records = Vec::new();
    let mut previous_log_time = None;
    for item in mcap::MessageStream::new(&bytes)? {
        let message = item?;
        if previous_log_time.is_some_and(|time| message.log_time < time) {
            bail!("measurement MCAP is not in arrival order");
        }
        previous_log_time = Some(message.log_time);
        let schema = message
            .channel
            .schema
            .as_ref()
            .map(|schema| schema.name.as_str())
            .ok_or_else(|| anyhow::anyhow!("measurement channel has no schema"))?;
        records.push(match schema {
            "fusion.ImuSample" => MeasurementRecord::Imu(ImuSample::decode(message.data.as_ref())?),
            "fusion.GpsFix" => MeasurementRecord::Gps(GpsFix::decode(message.data.as_ref())?),
            "fusion.CameraFrame" => {
                MeasurementRecord::Camera(CameraFrame::decode(message.data.as_ref())?)
            }
            "fusion.LidarScan" => {
                MeasurementRecord::Lidar(LidarScan::decode(message.data.as_ref())?)
            }
            other => bail!("unsupported measurement schema {other}"),
        });
    }
    Ok(records)
}

pub fn read_ego_truth(path: &Path) -> Result<Vec<EgoTruthState>> {
    read_schema(path, "fusion.EgoTruthState")
}

pub fn read_object_truth(path: &Path) -> Result<Vec<ObjectTruthState>> {
    read_schema(path, "fusion.ObjectTruthState")
}

pub fn read_imu_bias_truth(path: &Path) -> Result<Vec<ImuBiasTruth>> {
    read_schema(path, "fusion.ImuBiasTruth")
}

pub fn read_ego_estimates(path: &Path) -> Result<Vec<EgoStateEstimate>> {
    read_schema(path, "fusion.EgoStateEstimate")
}

pub fn read_tracks(path: &Path) -> Result<Vec<ObjectTrackFrame>> {
    read_schema(path, "fusion.ObjectTrackFrame")
}

pub(crate) fn write_tracker_history(
    output: &Path,
    name: &str,
    history: &TrackerHistory,
) -> Result<()> {
    fs::write(
        output
            .join("reports/baseline")
            .join(format!("tracker-history-{name}.json")),
        serde_json::to_vec_pretty(history)?,
    )?;
    Ok(())
}

pub(crate) fn read_tracker_history(path: &Path) -> Result<TrackerHistory> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("invalid {}", path.display()))
}

fn read_schema<T: Message + Default>(path: &Path, expected: &str) -> Result<Vec<T>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut values = Vec::new();
    for item in mcap::MessageStream::new(&bytes)? {
        let message = item?;
        if message
            .channel
            .schema
            .as_ref()
            .map(|schema| schema.name.as_str())
            == Some(expected)
        {
            values.push(T::decode(message.data.as_ref())?);
        }
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
pub fn write_reports(
    output: &Path,
    metrics: &RunMetrics,
    ego_timing: &TimingDiagnostics,
    gps_diagnostics: &GpsDiagnostics,
    tracker_estimated_diagnostics: &TrackerDiagnostics,
    tracker_truth_diagnostics: &TrackerDiagnostics,
    assumptions: &EstimatorAssumptions,
) -> Result<()> {
    let report_dir = output.join("reports/baseline");
    let json = serde_json::json!({
        "metrics": metrics,
        "ego_timing": ego_timing,
        "gps_fixes": gps_diagnostics,
        "estimated_ego_tracker_updates": tracker_estimated_diagnostics,
        "truth_ego_tracker_updates": tracker_truth_diagnostics,
        "ego_filter_assumptions": assumptions,
    });
    fs::write(
        report_dir.join("metrics.json"),
        serde_json::to_vec_pretty(&json)?,
    )?;
    let truth_ego_track_rmse = display_track_rmse(metrics.tracks_with_truth_ego.position_rmse_m);
    let estimated_ego_track_rmse =
        display_track_rmse(metrics.tracks_with_estimated_ego.position_rmse_m);
    let ego_cost = metrics
        .estimated_ego_position_rmse_delta_m
        .map(|value| format!("{value:+.3} m"))
        .unwrap_or_else(|| "—".to_owned());
    let mut summary = format!(
        "# Run result\n\n\
         GPS and IMU estimate the vehicle. Camera and lidar track objects.\n\n\
         ## Vehicle\n\n\
         Estimator: {}{}  \nPosition RMSE: {:.3} m  \nHeading RMSE: {:.3} rad  \nGPS fixes accepted/rejected/invalid: {}/{}/{}\n\n\
         ## Objects\n\n\
         Truth ego position RMSE: {truth_ego_track_rmse}  \nEstimated ego position RMSE: {estimated_ego_track_rmse}  \nCost of estimated ego: {ego_cost}\n\n\
         Estimated-ego associations, camera/lidar: {}/{}  \nUnmatched camera/lidar detections: {}/{}  \nTracks created/confirmed/deleted: {}/{}/{}\n",
        assumptions.algorithm.name(),
        assumptions
            .backend_version
            .as_ref()
            .map(|version| format!(" ({version})"))
            .unwrap_or_default(),
        metrics.ego.position_rmse_m,
        metrics.ego.yaw_rmse_rad,
        gps_diagnostics.accepted_fixes,
        gps_diagnostics.rejected_fixes,
        gps_diagnostics.invalid_fixes,
        tracker_estimated_diagnostics.associated_camera_detections,
        tracker_estimated_diagnostics.associated_lidar_detections,
        tracker_estimated_diagnostics.unmatched_camera_detections,
        tracker_estimated_diagnostics.unmatched_lidar_detections,
        tracker_estimated_diagnostics.created_tracks,
        tracker_estimated_diagnostics.confirmed_tracks,
        tracker_estimated_diagnostics.deleted_tracks,
    );
    summary.push_str(&format!(
        "Candidate pairs/gated out/selected: {}/{}/{}  \n\
         Missed/coasted tracker updates: {}/{}  \n\
         Estimated ego missed/false/switches/fragments: {}/{}/{}/{}  \n\
         Truth ego missed/false/switches/fragments: {}/{}/{}/{}\n",
        tracker_estimated_diagnostics.candidate_pairs,
        tracker_estimated_diagnostics.gated_out_pairs,
        tracker_estimated_diagnostics.selected_associations,
        tracker_estimated_diagnostics.missed_updates,
        tracker_estimated_diagnostics.coasted_updates,
        metrics.tracks_with_estimated_ego.missed_object_samples,
        metrics.tracks_with_estimated_ego.false_track_samples,
        metrics.tracks_with_estimated_ego.identity_switch_count,
        metrics.tracks_with_estimated_ego.track_fragment_count,
        metrics.tracks_with_truth_ego.missed_object_samples,
        metrics.tracks_with_truth_ego.false_track_samples,
        metrics.tracks_with_truth_ego.identity_switch_count,
        metrics.tracks_with_truth_ego.track_fragment_count,
    ));
    if let (Some(gyro_rmse), Some(accel_rmse), Some(gyro_coverage), Some(accel_coverage)) = (
        metrics.ego.gyro_bias_rmse_radps,
        metrics.ego.accel_bias_rmse_mps2,
        metrics.ego.gyro_bias_95pct_coverage,
        metrics.ego.accel_bias_95pct_coverage,
    ) {
        summary.push_str(&format!(
            "\n## IMU bias\n\n\
             Gyroscope bias RMSE: {gyro_rmse:.6} rad/s  \n\
             True gyroscope bias inside the 95% range: {:.1}%  \n\
             Accelerometer bias RMSE: {accel_rmse:.6} m/s²  \n\
             True accelerometer bias inside the 95% range: {:.1}%\n",
            gyro_coverage * 100.0,
            accel_coverage * 100.0,
        ));
    }
    if ego_timing.timing_compensation || ego_timing.maximum_delivery_age_ns > 0 {
        summary.push_str(&format!(
            "\n## Vehicle timing\n\n\
             Processing: {}  \n\
             Delayed measurements: {}  \n\
             Delayed measurements reordered: {}  \n\
             Revised estimates: {}  \n\
             Discarded measurements: {}  \n\
             Maximum arrival delay: {:.1} ms\n",
            if ego_timing.timing_compensation {
                "offline measurement-time order"
            } else {
                "arrival order"
            },
            ego_timing.delayed_measurements,
            ego_timing.replayed_measurements,
            ego_timing.revised_estimates,
            ego_timing.discarded_measurements,
            ego_timing.maximum_delivery_age_ns as f64 / 1_000_000.0,
        ));
    }
    summary.push_str("\n## Sensor freshness\n\nAges at the last GPS/IMU arrival; received packets, regardless of filter acceptance.\n\nSensor | Receipt age | Measurement age | Maximum receipt gap\n--- | --- | --- | ---\n");
    for (name, freshness) in [
        ("GPS", &ego_timing.gps_freshness),
        ("IMU", &ego_timing.imu_freshness),
    ] {
        let display_age = |age: Option<i64>| {
            age.map(|value| format!("{:.3} s", value as f64 / 1e9))
                .unwrap_or_else(|| "unavailable".to_owned())
        };
        summary.push_str(&format!(
            "{name} | {} | {} | {}\n",
            display_age(freshness.final_receipt_age_ns),
            display_age(freshness.final_measurement_age_ns),
            display_age(freshness.maximum_receipt_gap_ns),
        ));
    }
    summary.push_str("\nMaximum receipt gap includes trailing silence after the first packet. Unavailable means no packet was received.\n");
    fs::write(report_dir.join("summary.md"), summary)?;
    Ok(())
}

fn display_track_rmse(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3} m"))
        .unwrap_or_else(|| "— (no matched tracks)".to_owned())
}

fn write_measurements(path: &Path, records: &[MeasurementRecord]) -> Result<()> {
    let mut writer = new_writer(path)?;
    let mut channels = BTreeMap::new();
    for (schema_name, topic) in [
        ("fusion.ImuSample", "/measurement/imu"),
        ("fusion.GpsFix", "/measurement/gps"),
        ("fusion.CameraFrame", "/measurement/camera"),
        ("fusion.LidarScan", "/measurement/lidar"),
    ] {
        let schema = writer.add_schema(schema_name, "protobuf", FILE_DESCRIPTOR_SET)?;
        channels.insert(
            schema_name,
            writer.add_channel(schema, topic, "protobuf", &BTreeMap::new())?,
        );
    }
    for (sequence, record) in records.iter().enumerate() {
        let (schema, bytes) = match record {
            MeasurementRecord::Imu(value) => ("fusion.ImuSample", value.encode_to_vec()),
            MeasurementRecord::Gps(value) => ("fusion.GpsFix", value.encode_to_vec()),
            MeasurementRecord::Camera(value) => ("fusion.CameraFrame", value.encode_to_vec()),
            MeasurementRecord::Lidar(value) => ("fusion.LidarScan", value.encode_to_vec()),
        };
        let time = record.time();
        write_message(
            &mut writer,
            channels[schema],
            sequence as u32,
            time.arrival_time_ns,
            time.measurement_time_ns,
            &bytes,
        )?;
    }
    writer.finish()?;
    Ok(())
}

fn write_truth(
    path: &Path,
    ego: &[EgoTruthState],
    objects: &[ObjectTruthState],
    imu_bias: &[ImuBiasTruth],
) -> Result<()> {
    let mut writer = new_writer(path)?;
    let ego_schema = writer.add_schema("fusion.EgoTruthState", "protobuf", FILE_DESCRIPTOR_SET)?;
    let object_schema =
        writer.add_schema("fusion.ObjectTruthState", "protobuf", FILE_DESCRIPTOR_SET)?;
    let bias_schema = writer.add_schema("fusion.ImuBiasTruth", "protobuf", FILE_DESCRIPTOR_SET)?;
    let ego_channel = writer.add_channel(ego_schema, "/truth/ego", "protobuf", &BTreeMap::new())?;
    let object_channel = writer.add_channel(
        object_schema,
        "/truth/objects",
        "protobuf",
        &BTreeMap::new(),
    )?;
    let bias_channel =
        writer.add_channel(bias_schema, "/truth/imu_bias", "protobuf", &BTreeMap::new())?;
    for (sequence, state) in ego.iter().enumerate() {
        write_message(
            &mut writer,
            ego_channel,
            sequence as u32,
            state.time_ns,
            state.time_ns,
            &state.encode_to_vec(),
        )?;
    }
    for (sequence, state) in objects.iter().enumerate() {
        write_message(
            &mut writer,
            object_channel,
            sequence as u32,
            state.time_ns,
            state.time_ns,
            &state.encode_to_vec(),
        )?;
    }
    for (sequence, state) in imu_bias.iter().enumerate() {
        write_message(
            &mut writer,
            bias_channel,
            sequence as u32,
            state.time_ns,
            state.time_ns,
            &state.encode_to_vec(),
        )?;
    }
    writer.finish()?;
    Ok(())
}

fn new_writer(path: &Path) -> Result<Writer<BufWriter<File>>> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    Ok(Writer::new(BufWriter::new(file))?)
}

fn write_message<W: std::io::Write + std::io::Seek>(
    writer: &mut Writer<W>,
    channel_id: u16,
    sequence: u32,
    log_time_ns: i64,
    publish_time_ns: i64,
    data: &[u8],
) -> Result<()> {
    if log_time_ns < 0 || publish_time_ns < 0 {
        bail!("MCAP times must be nonnegative");
    }
    writer.write_to_known_channel(
        &MessageHeader {
            channel_id,
            sequence,
            log_time: log_time_ns as u64,
            publish_time: publish_time_ns as u64,
        },
        data,
    )?;
    Ok(())
}
