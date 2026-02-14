// Copyright (c) 2024-2026 SynerG Lab, IITB

#![allow(dead_code)]

mod monitor_metrics;
mod resource_manager_handler;
mod vcuda_client_handler;
#[path = "../common/mod.rs"]
mod common;



use std::thread;
use common::{config::CLMGR_CONFIG_PATH, utils::Utils};
use vcuda_client_handler::VCudaClientManager;
use monitor_metrics::VMMetrics;

use crate::resource_manager_handler::ResourceManagerHandler;

fn get_mqueue_path() -> String {
    let config = Utils::load_config_file(CLMGR_CONFIG_PATH);
    config["ipc"]["mqueue-path"].as_str().unwrap().to_string()
}

fn get_vcuda_process_monitor_period() -> u64 {
    let config = Utils::load_config_file(CLMGR_CONFIG_PATH);

    let period = || -> Option<u64> {
        Some(config.get("vcuda-client")?.get("process_monitor_period")?.as_integer()? as u64)
    };

    match period() {
        Some(p) => p as u64,
        None => 60u64
    }
}

fn get_resource_mgr_address() -> (String, u16) {
    let config = Utils::load_config_file(CLMGR_CONFIG_PATH);

    (config["resource-manager"]["address"].as_str().unwrap().to_string(), config["resource-manager"]["port"].as_integer().unwrap() as u16)
}


fn main() {

    env_logger::init();

    let (resource_manager_address, resource_manager_port) = get_resource_mgr_address();
    let mqueue_path = get_mqueue_path();

    let client_mgr = VCudaClientManager::new(&mqueue_path);
    let res_mgr = ResourceManagerHandler::new(resource_manager_address, resource_manager_port, &client_mgr);

    let (resource_manager_address, resource_manager_port) = VMMetrics::get_metrics_mgr_address();
    let address = format!("{}:{}", resource_manager_address, resource_manager_port);
    let shared_mem_path = VMMetrics::get_shared_memory_path();
    let mut metrics = VMMetrics::new(&client_mgr, shared_mem_path, &address);

    thread::scope(|s| {
        /*
        s.spawn(|| {
            let period = get_vcuda_process_monitor_period();
            loop {
                thread::sleep(std::time::Duration::from_secs(period));
                client_mgr.remove_closed_clients(|| res_mgr.notify_zero_clients());
            }
        });
        */

        s.spawn(|| {
            client_mgr.listen_to_clients(|gid, sm_core, rmgrflag | res_mgr.get_virt_server(s, gid, sm_core, rmgrflag ));
        });

        s.spawn(|| {
            metrics.start_monitor();
        });
    });
}
