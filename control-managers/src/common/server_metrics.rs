use std::{io::{self, BufRead, BufReader, Write}, net::TcpStream};
use byteorder::{BigEndian, ByteOrder};
use serde::{Serialize, Deserialize};
use serde_json;

const MAX_THREADS_PER_SM: u64 = 1536;
const WARP_SIZE: u64 = 32;
pub const METRICS_ARRAY_SIZE: usize = 10;
pub const MIN_METRICS_COUNT: usize = 5;

fn trend_based_prediction(resources: &[u64]) -> u64 {
    let mut deltas = Vec::new();

    // Calculate the delta between consecutive values
    for i in 1..resources.len() {
        deltas.push(resources[i] - resources[i - 1]);
    }

    // Average the deltas and add to the last difference
    let avg_delta: u64 = deltas.iter().sum::<u64>() / deltas.len() as u64;

    resources[resources.len() - 1] + avg_delta
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ServerMetricsInfo {
    threads: u64,
    thread_elapse_time_ms: u64,
    pub launch_count: u64,
    mem_usage: u64,
    mem_copy_size: u64,
    mem_cpy_elapse_time_ms: u64,
    memcpy_count: u64,
    total_streams: u64,
    total_events: u64,
}

impl ServerMetricsInfo {
    /// Parses a byte slice into a ServerMetricsInfo struct, handling big-endian conversion
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != std::mem::size_of::<ServerMetricsInfo>() {
            return Err(format!(
                "Incorrect byte size: expected {}, got {}",
                std::mem::size_of::<ServerMetricsInfo>(),
                bytes.len()
            ).into());
        }

        let threads = BigEndian::read_u64(&bytes[0..8]);
        let thread_elapse_time_ms = BigEndian::read_u64(&bytes[8..16]);
        let launch_count = BigEndian::read_u64(&bytes[16..24]);
        let mem_usage = BigEndian::read_u64(&bytes[24..32]);
        let mem_copy_size = BigEndian::read_u64(&bytes[32..40]);
        let mem_cpy_elapse_time_ms = BigEndian::read_u64(&bytes[40..48]);
        let memcpy_count = BigEndian::read_u64(&bytes[48..56]);
        let total_streams = BigEndian::read_u64(&bytes[56..64]);
        let total_events = BigEndian::read_u64(&bytes[64..72]);

        Ok(ServerMetricsInfo {
            threads,
            thread_elapse_time_ms,
            launch_count,
            mem_usage,
            mem_copy_size,
            mem_cpy_elapse_time_ms,
            memcpy_count,
            total_streams,
            total_events,
        })
    }

    pub fn parse_metrics_line(metrics_json: String) -> Option<ServerMetricsInfo> {
        let metrics_info: ServerMetricsInfo = serde_json::from_str(&metrics_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .ok()?; // Early return None if deserialization fails.

        Some(metrics_info)
    }

    // Method to calculate the average of the metrics
    fn calculate_average(resources: &[u64]) -> Option<u64> {
        if resources.is_empty() {
            return None;  // Return None if the metrics list is empty
        }

        let sum: u64 = resources.iter().sum();  // Sum of the metrics
        let count = resources.len();  // Number of metrics

        if count < MIN_METRICS_COUNT {
            return None;
        }

        Some(sum / count as u64)  // Return the average
    }

    fn convert_threads_warp(threads: u64) -> u32 {
        // for ub-11 and 12, max thread per SM = 1536
        let max_warps_per_sm = MAX_THREADS_PER_SM / WARP_SIZE;

        // Calculate total warps
        let total_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;

        // Estimate SMs occupied
        let mut sms_occupied = (total_warps + max_warps_per_sm - 1) / max_warps_per_sm;

        // Round to next even number
        if sms_occupied % 2!= 0{
            sms_occupied += 1;
        }

        sms_occupied as u32
    }

    pub fn predict_next_resource(metrics: &[ServerMetricsInfo], current_sm: u32, current_mem: u64) -> (u32,u64) {
        // Create arrays of mem_usage and threads
        let mem_usage_array: Vec<u64> = metrics.iter().map(|metric| metric.mem_usage as u64).collect();
        let threads_array: Vec<u64> = metrics.iter().map(|metric| metric.threads as u64).collect();

        let mem_predict: u64;
        let thread_predict: u64;
        let len = mem_usage_array.len();

        if len > MIN_METRICS_COUNT {
            mem_predict = trend_based_prediction(&mem_usage_array);
            thread_predict = trend_based_prediction(&threads_array);
        } else if len > 0 {
            mem_predict = *mem_usage_array.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0);
            thread_predict = *threads_array.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0);
        } else {
            return (current_sm, current_mem);
        }

        let sm_predict = Self::convert_threads_warp(thread_predict);
        (sm_predict, mem_predict)
    }

    pub fn write_server_metrics_info(&self, mut stream: &TcpStream) -> Option<()> {
        // Step 1: Serialize the data into a JSON string
        let serialized = match serde_json::to_string(self) {
            Ok(json) => json,
            Err(err) => {
                log::error!("Failed to serialize server metrics info: {}", err);
                // Write a failure response to the stream
                let error_message = format!("{{\"error\": \"Serialization failed: {}\"}}\n", err);
                if let Err(write_err) = stream.write_all(error_message.as_bytes()) {
                    log::error!("Failed to write error response to TCP stream: {}", write_err);
                }
                return None;
            }
        };

        // Step 2: Format the message with a newline
        let message = format!("{}\n", serialized);

        // Step 3: Write the message to the TCP stream
        if let Err(err) = stream.write_all(message.as_bytes()) {
            log::error!("Failed to write message to TCP stream: {}", err);
            return None;
        }

        // Step 4: Flush the stream to ensure data is sent immediately
        let _ = stream.flush();
        Some(())
    }

    pub fn read_server_metrics_info(stream: &TcpStream) -> Option<ServerMetricsInfo> {
        // Wrap TcpStream in BufReader
        let mut reader = BufReader::new(stream);

        // Use read_line
        let mut buffer = String::new();
        match reader.read_line(&mut buffer) {
            Ok(_) => {
                log::info!("Successfully read data from stream.");
            }
            Err(e) => {
                log::error!("Error reading from stream: {}", e);
                return None;
            }
        }

        // Try to deserialize the received data into ServerMetricsInfo
        match serde_json::from_slice::<ServerMetricsInfo>(buffer.as_bytes()) {
            Ok(client_info) => {
                log::info!("Successfully deserialized client metrics.");
                Some(client_info)
            }
            Err(e) => {
                log::error!("Error deserializing client metrics: {}", e);
                None
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClientMetricsInfo {
    pub gid: i32,
    pub latency: u64,
    pub upper_threshold: u64,
    pub lower_threshold: u64,
    pub latency_count: u64,
    pub latency_elapse_time_ms: u64,
}

impl ClientMetricsInfo {
    pub fn write_client_metrics_info(&self, mut stream: &TcpStream) -> Option<()> {
        // Step 1: Serialize the data into a JSON string
        let serialized = match serde_json::to_string(self) {
            Ok(json) => json,
            Err(err) => {
                log::error!("Failed to serialize server metrics info: {}", err);
                // Write a failure response to the stream
                let error_message = format!("{{\"error\": \"Serialization failed: {}\"}}\n", err);
                if let Err(write_err) = stream.write_all(error_message.as_bytes()) {
                    log::error!("Failed to write error response to TCP stream: {}", write_err);
                }
                return None;
            }
        };

        // Step 2: Format the message with a newline
        let message = format!("{}\n", serialized);

        // Step 3: Write the message to the TCP stream
        if let Err(err) = stream.write_all(message.as_bytes()) {
            log::error!("Failed to write message to TCP stream: {}", err);
            return None;
        }

        // Step 4: Flush the stream to ensure data is sent immediately
        let _ = stream.flush();
        Some(())
    }

    pub fn read_client_metrics_info(stream: &mut TcpStream) -> Option<ClientMetricsInfo> {
        // Wrap TcpStream in BufReader
        let mut reader = BufReader::new(stream);

        // Use read_line
        let mut buffer = String::new();
        match reader.read_line(&mut buffer) {
            Ok(_) => {
                log::info!("Successfully read data from stream.");
            }
            Err(e) => {
                log::error!("Error reading from stream: {}", e);
                return None;
            }
        }

        // Try to deserialize the received data into ClientMetricsInfo
        match serde_json::from_slice::<ClientMetricsInfo>(buffer.as_bytes()) {
            Ok(client_info) => {
                log::info!("Successfully deserialized client metrics.");
                Some(client_info)
            }
            Err(e) => {
                log::error!("Error deserializing client metrics: {}", e);
                None
            }
        }
    }

    pub fn predict_next_resource(metrics: &[ClientMetricsInfo], current_sm: u32, current_mem: u64) -> (u32, u64) {
        // Create arrays of metrics
        let latency_array: Vec<u64> = metrics.iter().map(|metric| metric.latency as u64).collect();
        let upper_array: Vec<u64> = metrics.iter().map(|metric| metric.upper_threshold as u64).collect();
        let lower_array: Vec<u64> = metrics.iter().map(|metric| metric.lower_threshold as u64).collect();

        let mut latency_predict = 0;
        let mut upper_predict = 0;
        let mut lower_predict = 0;
        let len = latency_array.len();

        if len > MIN_METRICS_COUNT {
            latency_predict = trend_based_prediction(&latency_array);
            upper_predict = trend_based_prediction(&upper_array);
            lower_predict = trend_based_prediction(&lower_array);
        } else if len > 0 {
            latency_predict = *latency_array.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0);
            upper_predict = *upper_array.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0);
            lower_predict = *lower_array.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0);
        }

        let mut sm_predict = current_sm;
        let mem_predict = current_mem;

        if lower_predict > 0 && latency_predict < lower_predict {
            // propotionate
            sm_predict = (sm_predict as u64 * latency_predict / lower_predict) as u32;
        }
        else if upper_predict > 0 && latency_predict > upper_predict {
            sm_predict = (sm_predict as u64 * latency_predict / upper_predict) as u32;
        }

        // Round to next even number
        if sm_predict % 2!= 0{
            sm_predict += 1;
        }
        (sm_predict, mem_predict)
    }
}
