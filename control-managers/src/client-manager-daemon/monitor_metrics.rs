#[path = "../common/mod.rs"]
mod common;

extern crate shmem;

use std::collections::{HashMap};
use std::sync::{Arc, Mutex};
use std::io::Read;
use std::fs::{OpenOptions, File};
use std::time::{SystemTime, UNIX_EPOCH};
use std::path::Path;
use nix::unistd;
use std::net::TcpStream;
use crate::vcuda_client_handler::VCudaClientManager;
use common::{config::CLMGR_CONFIG_PATH, utils::Utils};
use crate::common::server_metrics::ClientMetricsInfo;
use crate::common::server_metrics;

const METRICS_ARRAY_SIZE: usize = server_metrics::METRICS_ARRAY_SIZE;
const MIN_PERCENTAGE: f64 = 4.0;

// Struct to hold function-specific data.
#[derive(Debug, Clone)]
struct VMAppLatencyInfo {
    upper_threshold: u64,
    lower_threshold: u64,
    latency: u64,
    start_time: u64,
}

// Struct to hold app-specific data.
#[derive(Debug, Clone)]
struct VMAppMetricInfo {
    app_latency: Vec<VMAppLatencyInfo>,
    app_upper_threshold: u64,
    app_lower_threshold: u64,
    app_total_latency: u64,
    app_latency_count: u64,
    app_latency_start_index: u64,
    app_latency_start_time: u64,
    app_latency_end_time: u64,
    app_metrics_dirty: bool,
}

// Function to get the current time in milliseconds (equivalent to get_current_time)
fn get_current_time() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
}

// Struct to manage the metrics handler.
pub struct VMMetrics <'c> {
    shared_memory_file: String,
    file: Option<File>,
    app_metrics: Arc<Mutex<HashMap<u32, VMAppMetricInfo>>>,
    server_ipaddr: String,
    client_mgr: &'c VCudaClientManager,
}

impl <'c> VMMetrics <'c> {
    pub fn new(client_mgr: &'c VCudaClientManager, shared_memory_name: String, server_address: &String ) -> Self {
            println!("metrics constructor 3");
        if !Path::new(shared_memory_name.as_str()).exists() {
            unistd::mkfifo(shared_memory_name.as_str(), nix::sys::stat::Mode::S_IRWXU).expect("Failed to create FIFO");
        }
        VMMetrics {
            shared_memory_file: shared_memory_name.clone(),
            file: None,
            app_metrics: Arc::new(Mutex::new(HashMap::new())),
            server_ipaddr: server_address.clone(),
            client_mgr: client_mgr,

        }
    }

    fn insert_latency_record(&mut self, pid: u32, latency: u64, upperlimit: u64, lowerlimit: u64) {
        let mut app_metrics = self.app_metrics.lock().unwrap();

        // Check if the entry for the client exists, if not, create a new one
        let entry = app_metrics.entry(pid).or_insert(VMAppMetricInfo {
            app_latency: Vec::new(),
            app_upper_threshold: 0,
            app_lower_threshold: 0,
            app_total_latency: 0,
            app_latency_count: 0,
            app_latency_start_index: 0,
            app_latency_start_time: 0,
            app_latency_end_time: 0,
            app_metrics_dirty: false,
        });

        let write_index = (entry.app_latency_start_index + entry.app_latency_count) % METRICS_ARRAY_SIZE as u64; 

        // Handle cumulative totals and timestamps
        if entry.app_latency_count == METRICS_ARRAY_SIZE as u64 {
            // Subtract the resource of the oldest entry from the total
            entry.app_total_latency -= entry.app_latency[entry.app_latency_start_index as usize].latency;
            entry.app_upper_threshold -= entry.app_latency[entry.app_latency_start_index as usize].upper_threshold;
            entry.app_lower_threshold -= entry.app_latency[entry.app_latency_start_index as usize].lower_threshold;
            // Move the start index to the next oldest entry
            entry.app_latency_start_index = (entry.app_latency_start_index + 1) % 100;
        } else {
            entry.app_latency_count += 1;
        }

        // Get current time (you need to implement this function)
        let current_time = get_current_time();

        // Add the new resource usage to the buffer
        entry.app_latency.push(VMAppLatencyInfo {
            upper_threshold: upperlimit,
            lower_threshold: lowerlimit,
            latency: latency,
            start_time: current_time,
        });

        // Update cumulative totals and timestamps
        entry.app_total_latency += latency;
        entry.app_upper_threshold += upperlimit;
        entry.app_lower_threshold += lowerlimit;
        entry.app_latency_start_time = entry.app_latency[entry.app_latency_start_index as usize].start_time; // First entry
        entry.app_latency_end_time = entry.app_latency[write_index as usize].start_time; // Latest entry
        entry.app_metrics_dirty = true;

        // Log the data
        log::info!("Resource acquired: total_latency {} entries_count {} upper_threshold {} lower_threshold {} elapsed_time-ms {}",
                   entry.app_total_latency,
                   entry.app_latency_count,
                   entry.app_upper_threshold,
                   entry.app_lower_threshold,
                   (entry.app_latency_end_time - entry.app_latency_start_time) / 1000);
    }

    fn send_metrics_info(&self, stream: TcpStream) {
        //let client_ip = stream.peer_addr().unwrap().ip().to_string();

        let stream_clone = match stream.try_clone() {
            Ok(stream) => stream,
            Err(e) => {
                log::error!("Error cloning stream: {}", e);
                return;
            }
        };

        let mut app_metrics = self.app_metrics.lock().unwrap(); // Lock the Mutex

        // Loop through the HashMap entries
        for (client_id, metric_info) in app_metrics.iter_mut() {
            // `client_id` is the key, `metric_info` is the value
            if metric_info.app_metrics_dirty == false {
                continue;
            }

            metric_info.app_metrics_dirty = false;
            let gid = self.client_mgr.get_client_gid(*client_id);

            let client_metric_info = ClientMetricsInfo {
                gid: gid,
                latency: metric_info.app_total_latency,
                upper_threshold: metric_info.app_upper_threshold,
                lower_threshold: metric_info.app_lower_threshold,
                latency_count: metric_info.app_latency_count,
                latency_elapse_time_ms: (metric_info.app_latency_end_time - metric_info.app_latency_start_time) / 1000,
            };

            client_metric_info.write_client_metrics_info(&stream_clone);
        }
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
    fn handle_command(&mut self, metrics_info: &str) {
        let metric_lines: Vec<&str> = metrics_info.trim().split('\n').collect();
        for line in metric_lines {
            let mut parts = line.trim().split_whitespace();
            //println!("COMMAND: {:?}", parts);
            if let Some(metrics) = parts.next() {
                //println!("match metrics is {} {:?}", metrics, parts);
                match metrics {
                    "LATENCY" => {
                        let pid: u32 = parts.next().unwrap().parse().unwrap();
                        let latency: u64 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                        let upperlimit: u64 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                        let lowerlimit: u64 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                        {
                            self.insert_latency_record(pid, latency, upperlimit, lowerlimit);
                        }
                    }
                    "MEMORY" => {
                        let pid: u32 = parts.next().unwrap().parse().unwrap();
                        let memsize: u64 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                        let success: u32 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                        {
                            //TODO:: Track memory success/failure
                            //self.insert_latency_record(pid, latency, upperlimit, lowerlimit);
                        }
                    }
                    _ => {
                        println!("Unknown command received: {}", metrics);
                    }
                }
            }
        }
    }

    pub fn start_monitor(&mut self) {
        let fifo = OpenOptions::new()
            .read(true)
            .open(self.shared_memory_file.clone())
            .expect("Failed to open FIFO");
            println!("metrics constructor 5");

        self.file = Some(fifo);

        println!("After opening the file\n");

        //let stream = TcpStream::connect(self.server_ipaddr.clone());
        loop {
            if let Some(metrics_info) = self.read_from_shared_memory() {
                self.handle_command(&metrics_info);
            }
            /*
            let next_read_interval = self.find_next_read_interval();
            if !next_read_interval.is_zero() {
                thread::sleep(next_read_interval);
            }
            */
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

}
