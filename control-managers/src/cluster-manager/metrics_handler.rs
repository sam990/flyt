// Copyright (c) 2024-2026 SynerG Lab, IITB

////use std::sync::{Arc, Mutex};

use std::{thread, net::{TcpStream, TcpListener} };
use std::time::Duration;
use std::mem;
use std::cmp;


use crate::{client_handler::FlytClientManager, servernode_handler::ServerNodesManager};
use crate::common::server_metrics::{ServerMetricsInfo, ClientMetricsInfo};
use crate::common::server_metrics;
//use crate::metrics_handler::ResourceScaling;

const UPSCALE_COUNT: u32 = 3;
const DOWNSCALE_COUNT: u32 = 3;
const METRICS_ARRAY_SIZE: usize = server_metrics::METRICS_ARRAY_SIZE;

enum ChangeScale {
    ScaleUp,
    ScaleDown,
    ScaleNone,
}

#[derive(PartialEq)]
pub enum ResourceScaling {
    ResourceUnderAllocated,
    ResourceOverAllocated,
    ResourceNormalAllocated,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClientNodeMetrics {
    pub client_ipaddr: String,
    pub client_id: i32,
    pub server_ipaddr: String,
    pub rpc_id: u64,
    pub predicted_compute_units: u32,
    pub predicted_memory: u64,
    pub current_compute_units: u32,
    pub current_memory: u64,
}

impl ClientNodeMetrics {
    pub fn add_or_update_metric(
        &self,
        metrics: &mut Vec<ClientNodeMetrics>,
    ) {
        // Check if an entry with the same client_ipaddr and client_id exists
        if let Some(existing_metric) = metrics.iter_mut().find(|metric| {
            metric.client_ipaddr == self.client_ipaddr && metric.client_id == self.client_id
        }) {
            // Update the predicted_compute_units and predicted_memory to their maximum values
            existing_metric.predicted_compute_units =
                cmp::max(existing_metric.predicted_compute_units, self.predicted_compute_units);
            existing_metric.predicted_memory =
                cmp::max(existing_metric.predicted_memory, self.predicted_memory);
        } else {
            // If no entry exists, add the new metric to the vector
            metrics.push(self.clone());
        }
    }

    /// Function to determine resource scaling status
    pub fn check_resource_allocation_status(&self) -> ResourceScaling {
        let compute_status = if self.current_compute_units < self.predicted_compute_units {
            ResourceScaling::ResourceUnderAllocated
        } else if self.current_compute_units > self.predicted_compute_units {
            ResourceScaling::ResourceOverAllocated
        } else {
            ResourceScaling::ResourceNormalAllocated
        };

        let memory_status = if self.current_memory < self.predicted_memory {
            ResourceScaling::ResourceUnderAllocated
        } else if self.current_memory > self.predicted_memory {
            ResourceScaling::ResourceOverAllocated
        } else {
            ResourceScaling::ResourceNormalAllocated
        };

        // Combine statuses, prioritize under-allocation over over-allocation if one exists
        if matches!(compute_status, ResourceScaling::ResourceUnderAllocated)
            || matches!(memory_status, ResourceScaling::ResourceUnderAllocated)
        {
            ResourceScaling::ResourceUnderAllocated
        } else if matches!(compute_status, ResourceScaling::ResourceOverAllocated)
            || matches!(memory_status, ResourceScaling::ResourceOverAllocated)
        {
            ResourceScaling::ResourceOverAllocated
        } else {
            ResourceScaling::ResourceNormalAllocated
        }
    }
}

pub struct MetricsHandler<'a> {
    client_mgr: &'a FlytClientManager<'a>,
    server_nodes_manager: &'a ServerNodesManager<'a>,
}

impl <'a> MetricsHandler<'a> {
    pub fn new(client_mgr: &'a FlytClientManager, server_nodes_manager: &'a ServerNodesManager) -> Self {
        MetricsHandler {
            client_mgr,
            server_nodes_manager,
        }
    }

    pub fn start_metrics_handler(&self, port: u16) {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    self.handle_metrics_request(stream)
                }
                Err(e) => {
                    log::error!("Error accepting connection: {}", e)
                }
            }
        }
    }

    pub fn start_period_handler(&self, interval_secs: u64) {
        loop {
            // Sleep for the specified interval
            thread::sleep(Duration::from_secs(interval_secs));

            // Perform the network operation
            self.handle_heartbeat();
        }
    }

    fn handle_heartbeat(&self) {
        // For all server nodes, get the heartbeat metrics.
        // Compute scale up or down and take action.
        let mut clients = self.client_mgr.get_all_clients();
        let mut underallocated_list = Vec::<ClientNodeMetrics>::new();
        let mut overallocated_list = Vec::<ClientNodeMetrics>::new();

        log::info!("Heart beat start loop");
        for client_node in &mut clients {
            if *client_node.is_migrating.read().unwrap() == false  && client_node.virt_server.is_some() {
                let virt_server_ip = client_node.virt_server.as_ref().unwrap().read().unwrap().ipaddr.clone();
                let rpc_id = client_node.virt_server.as_ref().unwrap().read().unwrap().rpc_id;
                //let virt_server = virt_server.read().unwrap();
                let metrics = self.server_nodes_manager.get_server_node_metrics(&virt_server_ip, rpc_id);
                log::info!("Heart beat for server {} is {:?}", virt_server_ip, metrics);
                if let Some(metric) = metrics {

                    if metric.launch_count == 0 {
                        continue;
                    }

                    client_node.server_metrics.push(metric);
                    /* Predict the values */
                    let (sm_predict, mem_predict) = ServerMetricsInfo::predict_next_resource(&client_node.server_metrics, client_node.compute_requested, client_node.memory_requested);

                    let client_metrics = ClientNodeMetrics {
                        client_ipaddr           : client_node.ipaddr.clone(),
                        client_id               : client_node.client_id,
                        server_ipaddr           : virt_server_ip.clone(),
                        rpc_id                  : rpc_id,
                        current_compute_units   :client_node.compute_requested,
                        current_memory          :client_node.memory_requested,
                        predicted_compute_units : sm_predict,
                        predicted_memory        : mem_predict,
                        };

                    let resource_status = client_metrics.check_resource_allocation_status();
                    if resource_status == ResourceScaling::ResourceUnderAllocated {
                        client_metrics.add_or_update_metric(&mut underallocated_list);
                    }
                    else if resource_status == ResourceScaling::ResourceOverAllocated {
                        client_metrics.add_or_update_metric(&mut overallocated_list);
                    }

                    /* Remove the first elements to maintain a length of 15 */
                    if client_node.server_metrics.len() > METRICS_ARRAY_SIZE {
                        let x = client_node.server_metrics.len() - METRICS_ARRAY_SIZE;
                        client_node.server_metrics.drain(0..x);
                    }
                } else {
                    log::error!("Metrics is None, skipping...");
                    continue;
                }
            }
        }
        /* Release client */
        mem::drop(clients);

        /* Now we have under allocated and over allocated list */
        for overallocated_client in &overallocated_list {
            //if let Some(client_node) = self.client_mgr.get_client(&overallocated_client.client_ipaddr, overallocated_client.client_id) {
                //self.client_mgr.stop_client(&overallocated_client.client_ipaddr, overallocated_client.client_id);
                self.server_nodes_manager.change_resource_configurations(
                    &overallocated_client.server_ipaddr, 
                    overallocated_client.rpc_id, 
                    overallocated_client.predicted_compute_units, 
                    overallocated_client.predicted_memory, 
                    &overallocated_client.client_ipaddr,
                    );
                //self.client_mgr.resume_client(&overallocated_client.client_ipaddr, overallocated_client.client_id);
            //}
        }
        mem::drop(overallocated_list);

        /* Now we have under allocated and under allocated list */
        for underallocated_client in underallocated_list.clone() {
            //if let Some(client_node) = self.client_mgr.get_client(&underallocated_client.client_ipaddr, underallocated_client.client_id) {
                //self.client_mgr.stop_client(&underallocated_client.client_ipaddr, underallocated_client.client_id);
                let result = self.server_nodes_manager.change_resource_configurations(
                    &underallocated_client.server_ipaddr, 
                    underallocated_client.rpc_id, 
                    underallocated_client.predicted_compute_units, 
                    underallocated_client.predicted_memory, 
                    &underallocated_client.client_ipaddr,
                    );
                //self.client_mgr.resume_client(&underallocated_client.client_ipaddr, underallocated_client.client_id);

                if result.is_ok() {
                    // If the result is Ok, remove the underallocated client from the list
                    underallocated_list.retain(|client| client != &underallocated_client);
                }
            //}
        }

        if underallocated_list.len() == 0 {
            /* nothing to do */
            return;
        }

        /* We need to identify non-critical clients and downgrade them */
        // for now just log
        // TODO: implement this scenario
        // deallocate resource for non-critical applications or migrate non-critical migratable
        // applications and then vertical scale
        log::error!("GPU resource not avaible");
        for underallocated_client in underallocated_list {
            log::error!("underallocated client:  {:?}", underallocated_client);
        }
        
    }

    fn handle_metrics_request(&self, mut stream: TcpStream) {

        let client_ip = stream.peer_addr().unwrap().ip().to_string();

        if let Some(client_metric)  = ClientMetricsInfo::read_client_metrics_info(&mut stream) {
            if let Some(mut client_node) = self.client_mgr.get_client(&client_ip, client_metric.gid) {
                client_node.client_metrics.push(client_metric);
                let virt_server_ip = client_node.virt_server.as_ref().unwrap().read().unwrap().ipaddr.clone();
                let rpc_id = client_node.virt_server.as_ref().unwrap().read().unwrap().rpc_id;
                /* Predict the values */
                let (sm_predict, mem_predict) = ClientMetricsInfo::predict_next_resource(&client_node.client_metrics, client_node.compute_requested, client_node.memory_requested);

                let client_metrics = ClientNodeMetrics {
                    client_ipaddr           : client_ip.clone(),
                    client_id               : client_node.client_id,
                    server_ipaddr           : virt_server_ip.clone(),
                    rpc_id                  : rpc_id,
                    current_compute_units   :client_node.compute_requested,
                    current_memory          :client_node.memory_requested,
                    predicted_compute_units : sm_predict,
                    predicted_memory        : mem_predict,
                    };

                let resource_status = client_metrics.check_resource_allocation_status();
                if resource_status == ResourceScaling::ResourceUnderAllocated || 
                   resource_status == ResourceScaling::ResourceOverAllocated {

                    //self.client_mgr.stop_client(&client_metrics.client_ipaddr, client_metrics.client_id);
                    self.server_nodes_manager.change_resource_configurations(
                        &client_metrics.server_ipaddr, 
                        client_metrics.rpc_id, 
                        client_metrics.predicted_compute_units, 
                        client_metrics.predicted_memory, 
                        &client_metrics.client_ipaddr,
                        );
                    //self.client_mgr.resume_client(&client_metrics.client_ipaddr, client_metrics.client_id);
                }
                
                /* Remove the first elements to maintain a length of 15 */
                if client_node.client_metrics.len() > METRICS_ARRAY_SIZE {
                    let x = client_node.client_metrics.len() - METRICS_ARRAY_SIZE;
                    client_node.client_metrics.drain(0..x);
                }
            } else {
                log::error!("Client node is None, skipping...");
            }
        }
    }

}
