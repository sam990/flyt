#[path = "../common/mod.rs"]
mod common;

extern crate shmem;

use std::collections::{HashMap};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::io::{Read, Write};
use std::fs::{OpenOptions, File};
use std::path::Path;
use nix::unistd;
use std::net::TcpStream;
use crate::common::api_commands::MetricsCommand;
use crate::vcuda_client_handler::VCudaClientManager;
use common::{config::CLMGR_CONFIG_PATH, utils::Utils};

const MIN_PERCENTAGE: f64 = 4.0;

// Struct to hold function-specific data.
#[derive(Debug, Clone)]
struct FunctionData {
    name: String,
    upper_threshold: f64,
    lower_threshold: f64,
    backoff: usize,
    spike_count: usize,
    metrics: Vec<f64>,
}

// Struct to hold PID-specific data.
#[derive(Debug)]
struct PIDData {
    functions: HashMap<String, FunctionData>,
    read_interval: u128,
    resource_delta: Vec<f64>,
    last_read_time: Instant,
}

// Struct to manage the metrics handler.
pub struct Metrics <'c> {
    shared_memory_file: String,
    file: Option<File>,
    pids: Arc<Mutex<HashMap<u32, PIDData>>>,
    default_time_interval: u64,
    address: String,
    scaleupfactor: u16,
    scaledownfactor: u16,
    client_mgr: &'c VCudaClientManager,
}

impl <'c> Metrics <'c> {
    pub fn new(client_mgr: &'c VCudaClientManager, default_interval: u64, shared_memory_name: String, server_address: &String, scaleup_factor: u16, scaledown_factor: u16) -> Self {
            println!("metrics constructor 3");
        if !Path::new(shared_memory_name.as_str()).exists() {
            unistd::mkfifo(shared_memory_name.as_str(), nix::sys::stat::Mode::S_IRWXU).expect("Failed to create FIFO");
        }
            println!("metrics constructor {} {} {} ", default_interval, scaleup_factor, scaledown_factor);
        Metrics {
            shared_memory_file: shared_memory_name.clone(),
            file: None,
            pids: Arc::new(Mutex::new(HashMap::new())),
            default_time_interval: default_interval,
            address: server_address.clone(),
            scaleupfactor: scaleup_factor,
            scaledownfactor: scaledown_factor,
            client_mgr: client_mgr,

        }
    }

    fn handle_start_command(&self, pid: u32, read_interval: u64) {
            log::info!("metrics start command");
            println!("metrics start command");
        let mut pids = self.pids.lock().unwrap();
        pids.entry(pid)
            .or_insert(PIDData {
                functions: HashMap::new(),
                read_interval: read_interval.into(),
                resource_delta: Vec::new(),
                last_read_time: Instant::now(),
            });
    }

    fn handle_fn_init_command(
        &self,
        pid: u32,
        function_name: String,
        upper_threshold: f64,
        lower_threshold: f64,
        spike_count: usize,
    ) {
            log::info!("metrics init command");
            println!("metrics init command");
        let mut pids = self.pids.lock().unwrap();
        if let Some(pid_data) = pids.get_mut(&pid) {
            pid_data.functions.insert(
                function_name.clone(),
                FunctionData {
                    name: function_name,
                    upper_threshold,
                    lower_threshold,
                    backoff: 0,
                    spike_count,
                    metrics: Vec::new(),
                },
            );
        }
    }

    fn handle_log_command(&mut self, pid: u32, function_name: String, metric: f64) {
        let mut pids = self.pids.lock().unwrap();
        if let Some(pid_data) = pids.get_mut(&pid) {
            pid_data.resource_delta.push(metric);
            /*
            if let Some(function_data) = pid_data.functions.get_mut(&function_name) {
                /* Have scaleup recently, so backoff.. */
                if function_data.backoff > 0 {
                    function_data.backoff -= 1;
                    log::info!("bakcing off {}\n", function_data.backoff);
                    return;
                }

                function_data.metrics.push(metric);
                if function_data.metrics.len() > function_data.spike_count {
                    function_data.metrics.remove(0);
                }

                let all_above = function_data.metrics.iter().all(|&m| m > function_data.upper_threshold);
                let all_below = function_data.metrics.iter().all(|&m| m < function_data.lower_threshold);
                log::info!("metrics log command threshold: {}, all_above: {} all_below: {}", function_data.upper_threshold, all_above, all_below);

                if all_above {
                    self.send_scale_command(pid, true);
                    function_data.metrics.clear();
                    function_data.backoff = function_data.spike_count;
                } else if all_below {
                    self.send_scale_command(pid, false);
                    function_data.metrics.clear();
                    function_data.backoff = function_data.spike_count;
                }
            }
        */
            pid_data.resource_delta.push(metric);
            let avg = if pid_data.resource_delta.len() > 5 && pid_data.resource_delta.len() < 10 {
                calculate_average(&pid_data.resource_delta)
            }
            else if pid_data.resource_delta.len() >= 10 {
                Some(predict_next_diff(&pid_data.resource_delta))
            }
            else {
                None
            };

            /* Remove the first elements to maintain a length of 15 */
            let x = pid_data.resource_delta.len() - 15;
            // Remove the first `x` elements
            if x > 0 && x <= pid_data.resource_delta.len() {
                pid_data.resource_delta.drain(0..x);
            }

            log::info!("Predicted pid: {} scale value {} for metric: {}", pid, avg.unwrap_or(0.0), metric);
            match avg {
                Some(avg_value) => {

                    // Check if the absolute difference exceeds the threshold
                    if avg_value.abs() > MIN_PERCENTAGE {
                        if avg_value > 0.0 {
                            self.scaleupfactor = avg_value.abs() as u16;
                            self.send_scale_command(pid, true);
                        } else {
                            self.scaledownfactor = avg_value.abs() as u16;
                            self.send_scale_command(pid, false);
                        }
                    } 
                }
                None => {
                }
            }
        }
    }

    fn handle_stop_command(&self, pid: u32) {
            log::info!("metrics stop command");
            println!("metrics stop command");
        let mut pids = self.pids.lock().unwrap();
        pids.remove(&pid);
    }

    fn send_scale_command(&self, pid: u32, upscale: bool) {
        let stream = TcpStream::connect(self.address.clone());

        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                log::error!("Error connecting to server: {}", error);
                return;
            }
        };

        let client_id = self.client_mgr.get_client_gid(pid);

        let message = if upscale { 
            format!( "{}\n{},{}\n", MetricsCommand::CLIENTD_MMGR_UPSCALE, client_id, self.scaleupfactor)
        } else {
            format!( "{}\n{},{}\n", MetricsCommand::CLIENTD_MMGR_DOWNSCALE, client_id, self.scaledownfactor)
        };

        if let Err(e) = stream.write_all(message.as_bytes()) {
            log::error!("Error writing to stream: {}", e);
            return;
        }

        log::debug!("Scaling resource upscale: {}", message);
    }

    fn read_from_shared_memory(&mut self) -> Option<String> {
        // Map the shared memory to a slice of bytes
        let mut buffer = String::new();
        if let Some(ref mut f) = self.file {
            f.read_to_string(&mut buffer).expect("Failed to read from FIFO");
        }
        else {
            return None;
        }
        Some(buffer)
    }
    fn handle_command(&mut self, commands: &str) {
        let cmdlines: Vec<&str> = commands.trim().split('\n').collect();
        for line in cmdlines {
            let mut parts = line.trim().split_whitespace();
            //println!("COMMAND: {:?}", parts);
            if let Some(command) = parts.next() {
                //println!("match command is {} {:?}", command, parts);
                match command {
                    "START" => {
                        let pid: u32 = parts.next().unwrap().parse().unwrap();
                        let interval: u64 = parts.next().map_or(self.default_time_interval, |s| s.parse().unwrap_or(self.default_time_interval));
                        {
                            self.handle_start_command(pid, interval);
                        }
                        println!("START command received for PID {} with interval {}", pid, interval);
                    }
                    "FN_INIT" => {
                        let pid: u32 = parts.next().unwrap().parse().unwrap();
                        let func_name = parts.next().unwrap().to_string();
                        let upper_threshold: f64 = parts.next().unwrap().parse().unwrap();
                        let lower_threshold: f64 = parts.next().unwrap().parse().unwrap();
                        let spike_count: usize = parts.next().unwrap().parse().unwrap();
                        {
                            self.handle_fn_init_command(pid, func_name.clone(), upper_threshold, lower_threshold, spike_count);
                        }
                        println!(
                            "FN_INIT command received for PID {} and function {}",
                            pid, func_name
                        );

                    }
                    "LOG" => {
                        let pid: u32 = parts.next().unwrap().parse().unwrap();
                        let func_name = parts.next().unwrap().to_string();
                        let metric: u32 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                        {
                            self.handle_log_command(pid, func_name, metric as f64);
                        }
                    }
                    "STOP" => {
                        let pid: u32 = parts.next().unwrap().parse().unwrap();
                        {
                            self.handle_stop_command(pid);
                        }
                        println!("STOP command received for PID {}", pid);
                    }
                    _ => {
                        println!("Unknown command received: {}", command);
                    }
                }
            }
        }
    }
    fn read_pid_data(&self, pid: u32) {
        // Logic to read and process data for the given PID
        println!("Reading data for PID: {}", pid);

        // Update the last read time for the PID
        let mut pids = self.pids.lock().unwrap();
        if let Some(pid_data) = pids.get_mut(&pid) {
            pid_data.last_read_time = Instant::now();
        }
    }

    fn find_next_read_interval(&self) -> Duration {
        let pids_data = self.pids.lock().unwrap();
        let now = Instant::now();

        // Calculate the remaining time for each PID
        pids_data
            .values()  // Iterate over the values in the HashMap (PIDData)
            .map(|pid_data| {
                let elapsed = now.duration_since(pid_data.last_read_time);
                if elapsed.as_millis() >= pid_data.read_interval {
                    Duration::ZERO
                } else {
                    Duration::from_millis((pid_data.read_interval - elapsed.as_millis()) as u64)
                }
            })
            .min()  // Find the minimum duration
            .unwrap_or(Duration::from_millis(self.default_time_interval))  // Return the minimum or default interval
    }

    pub fn start_monitor(&mut self) {
        let fifo = OpenOptions::new()
            .read(true)
            .open(self.shared_memory_file.clone())
            .expect("Failed to open FIFO");
            println!("metrics constructor 5");

        self.file = Some(fifo);

        println!("After opening the file\n");
        loop {
            if let Some(command) = self.read_from_shared_memory() {
                self.handle_command(&command);
            }
            let next_read_interval = self.find_next_read_interval();
            if !next_read_interval.is_zero() {
                thread::sleep(next_read_interval);
            }
        }
    }

    pub fn get_metrics_mgr_address() -> (String, u16) {
        let config = Utils::load_config_file(CLMGR_CONFIG_PATH);

        (config["resource-manager"]["address"].as_str().unwrap().to_string(), 
         config["resource-manager"]["metrics-port"].as_integer().unwrap() as u16)
    }

    pub fn get_shared_memory_path() -> String {
        let config = Utils::load_config_file(CLMGR_CONFIG_PATH);

        return config["resource-metrics"]["log-file"].as_str().unwrap().to_string();
    }

    pub fn get_metric_thresholds() -> (u16, u16, u16) {
        let config = Utils::load_config_file(CLMGR_CONFIG_PATH);

        (config["resource-metrics"]["scaleup-factor"].as_integer().unwrap() as u16, 
         config["resource-metrics"]["scaledown-factor"].as_integer().unwrap() as u16, 
         config["resource-metrics"]["metric-interval"].as_integer().unwrap() as u16)
    }
}
    // Method to calculate the average of the metrics
    fn calculate_average(resources: &[f64]) -> Option<f64> {
        if resources.is_empty() {
            return None;  // Return None if the metrics list is empty
        }

        let sum: f64 = resources.iter().sum();  // Sum of the metrics
        let count = resources.len();  // Number of metrics

        if count < 5 {
            return None;
        }

        Some(sum / count as f64)  // Return the average
    }

    fn trend_based_prediction(diffs: &[f64]) -> f64 {
        let mut deltas = Vec::new();

        // Calculate the delta between consecutive values
        for i in 1..diffs.len() {
            deltas.push(diffs[i] - diffs[i - 1]);
        }

        // Average the deltas and add to the last difference
        let avg_delta: f64 = deltas.iter().sum::<f64>() / deltas.len() as f64;

        diffs[diffs.len() - 1] + avg_delta
    }

    fn weighted_average(diffs: &[f64]) -> f64 {
        let weights: Vec<usize> = (1..=diffs.len()).collect(); // Weights for each difference (latest has higher weight)
        let weighted_sum: f64 = diffs.iter()
                                     .zip(weights.iter())
                                     .map(|(diff, weight)| diff * (*weight as f64))
                                     .sum();

        let total_weights: i32 = weights.iter().map(|&w| w as i32).sum();

        weighted_sum / total_weights as f64
    }

    fn predict_next_diff(diffs: &[f64]) -> f64 {
         (trend_based_prediction(diffs) + weighted_average(diffs))/2.0
    }


/*
impl <'c> Drop for Metrics <'c> {
    fn drop(&mut self) {
        println!("Dropping FileManager and closing file");
        let _ = std::fs::remove_file(self.shared_memory_file.clone());
        // The file is automatically closed when FileManager is dropped
        // Additional cleanup can be performed here if needed
    }
}
*/

/*
fn main() {
    let shared_memory_path = String::from("/path/to/shared/memory");
    let metrics_handler = Metrics::new(60, shared_memory_path);

    // Example of starting the metrics handler in a separate thread
    let metrics_handler_thread = thread::spawn(move || {
        metrics_handler.run();
    });

    // Wait for the thread to finish
    metrics_handler_thread.join().unwrap();
}
*/
