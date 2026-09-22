pub mod bundle;
pub mod compare;
pub mod estimator;
pub mod eval;
pub mod external;
pub mod math;
pub mod pipeline;
pub mod random;
pub mod scenario;
pub mod sensor;
pub mod sweep;
pub mod tracker;
pub mod truth;
pub mod viz;

use std::path::{Path, PathBuf};

use crate::{
    bundle::{GeneratedRun, MeasurementRecord},
    estimator::EgoMeasurement,
    scenario::ResolvedScenario,
    tracker::{EgoHistory, EgoSource, PerceptionMeasurement},
};
use anyhow::Result;

pub fn resolve_scenario(path: &Path) -> Result<ResolvedScenario> {
    scenario::load_and_resolve(path)
}

pub fn run_experiment(scenario_path: &Path, output: &Path) -> Result<PathBuf> {
    let scenario = resolve_scenario(scenario_path)?;
    run_resolved_experiment(&scenario, output, true)?;
    Ok(output.to_path_buf())
}

pub fn run_numbered_experiment(scenario_path: &Path, output: Option<&Path>) -> Result<PathBuf> {
    let output = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| next_run_path(Path::new("runs")));
    run_experiment(scenario_path, &output)
}

pub fn next_run_path(root: &Path) -> PathBuf {
    for number in 1_u64.. {
        let candidate = root.join(format!("run{number:03}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("run number overflow")
}

pub(crate) fn run_resolved_experiment(
    scenario: &ResolvedScenario,
    output: &Path,
    build_visualization: bool,
) -> Result<eval::RunMetrics> {
    bundle::prepare(output, scenario)?;
    let generated = sensor::generate(scenario)?;
    let ego_measurements = generated
        .measurements
        .iter()
        .filter_map(|record| match record {
            MeasurementRecord::Imu(value) => Some(EgoMeasurement::Imu(*value)),
            MeasurementRecord::Gps(value) => Some(EgoMeasurement::Gps(*value)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let perception_measurements = generated
        .measurements
        .iter()
        .filter_map(|record| match record {
            MeasurementRecord::Camera(value) => Some(PerceptionMeasurement::Camera(value.clone())),
            MeasurementRecord::Lidar(value) => Some(PerceptionMeasurement::Lidar(value.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();

    let ego_run = estimator::run(&scenario.ego_estimator, &scenario.imu, &ego_measurements)?;
    let estimated_history = EgoHistory::from_estimates(&ego_run.estimates)?;
    let estimated_tracker = tracker::run(
        &scenario.object_tracker,
        &scenario.camera,
        &scenario.lidar,
        &perception_measurements,
        &estimated_history,
    )?;

    let truth_history = EgoHistory::from_truth(&generated.ego_truth_states)?;
    let truth_tracker = tracker::run(
        &scenario.object_tracker,
        &scenario.camera,
        &scenario.lidar,
        &perception_measurements,
        &truth_history,
    )?;
    anyhow::ensure!(
        estimated_tracker.processed_detections == truth_tracker.processed_detections,
        "tracker comparison did not use identical detections"
    );

    let metrics = eval::evaluate(
        scenario,
        &generated.ego_truth_states,
        &generated.object_truth_states,
        &generated.imu_bias_truth,
        &ego_run.estimates,
        &estimated_tracker.frames,
        &truth_tracker.frames,
    );

    bundle::write_generated(output, &generated)?;
    bundle::write_ego_estimates(output, &ego_run.estimates)?;
    bundle::write_tracks(output, "estimated-ego", &estimated_tracker.frames)?;
    bundle::write_tracks(output, "truth-ego", &truth_tracker.frames)?;
    bundle::write_tracker_history(output, "estimated-ego", &estimated_tracker.history)?;
    bundle::write_tracker_history(output, "truth-ego", &truth_tracker.history)?;
    bundle::write_reports(
        output,
        &metrics,
        &ego_run.timing,
        &ego_run.gps_diagnostics,
        &estimated_tracker.diagnostics,
        &truth_tracker.diagnostics,
        &ego_run.assumptions,
    )?;
    if build_visualization {
        viz::write_bundle_visualization(output, &viz::default_visualization_path(output))?;
    }
    Ok(metrics)
}

pub fn generate_bundle(scenario_path: &Path, output: &Path) -> Result<GeneratedRun> {
    let scenario = resolve_scenario(scenario_path)?;
    bundle::prepare(output, &scenario)?;
    let generated = sensor::generate(&scenario)?;
    bundle::write_generated(output, &generated)?;
    Ok(generated)
}

pub fn score_ego_csv(run: &Path, csv: &Path, id: &str) -> Result<eval::EgoMetrics> {
    validate_external_id(id)?;
    let scenario = resolve_scenario(&run.join("scenario.resolved.yaml"))?;
    let estimates = external::read_ego_csv(csv, id, "world", "body")?;
    let relative = format!("estimates/{id}.mcap");
    anyhow::ensure!(
        !run.join(&relative).exists(),
        "{} already exists",
        run.join(&relative).display()
    );
    let truth_path = run.join("truth.mcap");
    let metrics = eval::evaluate_ego(
        &scenario,
        &bundle::read_ego_truth(&truth_path)?,
        &bundle::read_imu_bias_truth(&truth_path)?,
        &estimates,
    );
    anyhow::ensure!(
        metrics.matched_samples > 0,
        "cannot score ego output: 0 of {} estimates matched truth within {} ms",
        metrics.estimate_samples,
        scenario.metrics.max_truth_match_gap_ns as f64 / 1_000_000.0
    );
    bundle::write_ego_estimates_file(&run.join(&relative), id, &estimates)?;
    write_external_report(run, id, &metrics)?;
    Ok(metrics)
}

pub fn score_tracks_csv(
    run: &Path,
    csv: &Path,
    id: &str,
    ego_source: EgoSource,
) -> Result<eval::TrackMetrics> {
    validate_external_id(id)?;
    let scenario = resolve_scenario(&run.join("scenario.resolved.yaml"))?;
    let tracks = external::read_tracks_csv(csv, id, "world", ego_source)?;
    let relative = format!("tracks/{id}.mcap");
    anyhow::ensure!(
        !run.join(&relative).exists(),
        "{} already exists",
        run.join(&relative).display()
    );
    let truth_path = run.join("truth.mcap");
    let metrics = eval::evaluate_tracks(
        &scenario,
        &bundle::read_ego_truth(&truth_path)?,
        &bundle::read_object_truth(&truth_path)?,
        &bundle::read_ego_estimates(&run.join("estimates/ego-baseline.mcap"))?,
        &tracks,
        ego_source,
    );
    anyhow::ensure!(
        metrics.matched_samples > 0,
        "cannot score track output: 0 of {} tracks matched truth within {} ms",
        metrics.track_samples,
        scenario.metrics.max_truth_match_gap_ns as f64 / 1_000_000.0
    );
    bundle::write_tracks_file(&run.join(&relative), id, &tracks)?;
    write_external_report(run, id, &metrics)?;
    Ok(metrics)
}

fn validate_external_id(id: &str) -> Result<()> {
    anyhow::ensure!(
        !id.is_empty()
            && id.chars().all(
                |character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            ),
        "ID must contain only letters, numbers, '-' and '_'"
    );
    Ok(())
}

fn write_external_report(run: &Path, id: &str, metrics: &impl serde::Serialize) -> Result<()> {
    let directory = run.join("reports").join(id);
    std::fs::create_dir_all(&directory)?;
    std::fs::write(
        directory.join("metrics.json"),
        serde_json::to_vec_pretty(metrics)?,
    )?;
    Ok(())
}
