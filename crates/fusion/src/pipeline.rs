use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorKind {
    Imu,
    Gps,
    Camera,
    Lidar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineEvent {
    pub sensor: SensorKind,
    pub arrival_time_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DropPolicy {
    DropNewest,
    DropOldest,
    KeepLatest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineConfig {
    pub queue_capacity: usize,
    pub drop_policy: DropPolicy,
    pub imu_processing_ns: i64,
    pub gps_processing_ns: i64,
    pub camera_processing_ns: i64,
    pub lidar_processing_ns: i64,
}

impl PipelineConfig {
    fn processing_time_ns(self, sensor: SensorKind) -> i64 {
        match sensor {
            SensorKind::Imu => self.imu_processing_ns,
            SensorKind::Gps => self.gps_processing_ns,
            SensorKind::Camera => self.camera_processing_ns,
            SensorKind::Lidar => self.lidar_processing_ns,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineReport {
    pub processed_events: usize,
    pub dropped_events: usize,
    pub maximum_queue_depth: usize,
    pub total_queue_wait_ns: i64,
    pub maximum_message_age_ns: i64,
}

pub fn replay(events: &[PipelineEvent], config: PipelineConfig) -> PipelineReport {
    let mut queue = VecDeque::new();
    let mut report = PipelineReport::default();
    let mut now_ns = 0;
    let mut index = 0;

    while index < events.len() || !queue.is_empty() {
        while index < events.len() && events[index].arrival_time_ns <= now_ns {
            if config.queue_capacity == 0 {
                report.dropped_events += 1;
            } else if queue.len() == config.queue_capacity {
                match config.drop_policy {
                    DropPolicy::DropNewest => report.dropped_events += 1,
                    DropPolicy::DropOldest => {
                        queue.pop_front();
                        queue.push_back(events[index]);
                        report.dropped_events += 1;
                    }
                    DropPolicy::KeepLatest => {
                        report.dropped_events += queue.len();
                        queue.clear();
                        queue.push_back(events[index]);
                    }
                }
            } else {
                queue.push_back(events[index]);
                report.maximum_queue_depth = report.maximum_queue_depth.max(queue.len());
            }
            index += 1;
        }

        if queue.is_empty() {
            now_ns = events[index].arrival_time_ns;
            continue;
        }

        let event = queue.pop_front().expect("queue is not empty");
        let start_ns = now_ns.max(event.arrival_time_ns);
        let wait_ns = start_ns - event.arrival_time_ns;
        report.processed_events += 1;
        report.total_queue_wait_ns += wait_ns;
        now_ns = start_ns + config.processing_time_ns(event.sensor);
        report.maximum_message_age_ns = report
            .maximum_message_age_ns
            .max(now_ns - event.arrival_time_ns);
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(
        queue_capacity: usize,
        processing_ns: i64,
        drop_policy: DropPolicy,
    ) -> PipelineConfig {
        PipelineConfig {
            queue_capacity,
            drop_policy,
            imu_processing_ns: processing_ns,
            gps_processing_ns: processing_ns,
            camera_processing_ns: processing_ns,
            lidar_processing_ns: processing_ns,
        }
    }

    #[test]
    fn idle_pipeline_has_no_wait() {
        let events = [
            PipelineEvent {
                sensor: SensorKind::Imu,
                arrival_time_ns: 0,
            },
            PipelineEvent {
                sensor: SensorKind::Imu,
                arrival_time_ns: 10,
            },
        ];
        let report = replay(&events, config(4, 5, DropPolicy::DropNewest));
        assert_eq!(report.processed_events, 2);
        assert_eq!(report.dropped_events, 0);
        assert_eq!(report.total_queue_wait_ns, 0);
        assert_eq!(report.maximum_message_age_ns, 5);
    }

    #[test]
    fn a_burst_drops_new_events_when_full() {
        let events = [
            PipelineEvent {
                sensor: SensorKind::Lidar,
                arrival_time_ns: 0,
            },
            PipelineEvent {
                sensor: SensorKind::Lidar,
                arrival_time_ns: 0,
            },
            PipelineEvent {
                sensor: SensorKind::Lidar,
                arrival_time_ns: 0,
            },
        ];
        let report = replay(&events, config(1, 10, DropPolicy::DropNewest));
        assert_eq!(report.processed_events, 1);
        assert_eq!(report.dropped_events, 2);
        assert_eq!(report.maximum_queue_depth, 1);
        assert_eq!(report.total_queue_wait_ns, 0);
    }

    #[test]
    fn replay_is_deterministic() {
        let events = [
            PipelineEvent {
                sensor: SensorKind::Camera,
                arrival_time_ns: 0,
            },
            PipelineEvent {
                sensor: SensorKind::Imu,
                arrival_time_ns: 2,
            },
        ];
        let first = replay(&events, config(2, 7, DropPolicy::DropNewest));
        assert_eq!(first, replay(&events, config(2, 7, DropPolicy::DropNewest)));
    }

    #[test]
    fn drop_oldest_keeps_the_new_arrival() {
        let events = [
            PipelineEvent {
                sensor: SensorKind::Gps,
                arrival_time_ns: 0,
            },
            PipelineEvent {
                sensor: SensorKind::Gps,
                arrival_time_ns: 0,
            },
        ];
        let report = replay(&events, config(1, 10, DropPolicy::DropOldest));
        assert_eq!(report.processed_events, 1);
        assert_eq!(report.dropped_events, 1);
    }

    #[test]
    fn keep_latest_discards_the_backlog() {
        let events = [
            PipelineEvent {
                sensor: SensorKind::Camera,
                arrival_time_ns: 0,
            },
            PipelineEvent {
                sensor: SensorKind::Camera,
                arrival_time_ns: 0,
            },
            PipelineEvent {
                sensor: SensorKind::Camera,
                arrival_time_ns: 0,
            },
        ];
        let report = replay(&events, config(1, 10, DropPolicy::KeepLatest));
        assert_eq!(report.processed_events, 1);
        assert_eq!(report.dropped_events, 2);
        assert_eq!(report.total_queue_wait_ns, 0);
    }
}
