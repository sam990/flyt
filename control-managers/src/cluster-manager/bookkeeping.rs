// Copyright (c) 2024-2026 SynerG Lab, IITB

use std::sync::RwLock;
use std::sync::Arc;
use std::net::TcpStream;
use toml::Table;
use mongodb::{options::{ClientOptions, ServerAddress, Credential, FindOneAndUpdateOptions, ReturnDocument}, sync::Client, sync::Collection, bson::doc};
use serde::{Deserialize, Serialize};

use crate::common::config::RMGR_CONFIG_PATH;
use crate::common::types::StreamEnds;
use crate::common::utils::Utils;

struct ConfigOptions;

impl ConfigOptions {
    const VIRT_SERVER_DEALLOCATE_TIME: RwLock<Option<Option<u64>>> = RwLock::new(None);
}

#[derive(Debug, Clone)]
pub struct GPU {
    pub name: String,
    pub memory: u64,
    pub compute_units: u32,
    pub compute_power: u64,
    pub gpu_id: u64,
    pub allocated_memory: u64,
    pub allocated_compute_units: u32,
    pub actual_allocated_units: u32,
}

impl Default for GPU {
    fn default() -> Self {
        GPU {
            name: "".to_string(),
            memory: 0,
            compute_units: 0,
            compute_power: 0,
            gpu_id: 0,
            allocated_memory: 0,
            allocated_compute_units: 0,
            actual_allocated_units : 0,
        }
    }
}

const DEFAULT_ADD_LOAD: i32 = 5;

pub struct ServerNode {
    pub ipaddr: String,
    pub gpus: Vec<Arc<RwLock<GPU>>>,
    pub stream: Arc<RwLock<StreamEnds<TcpStream>>>,
    pub virt_servers: Vec<Arc<RwLock<VirtServer>>>,
    pub metrics: Option<Vec<u32>>,
    pub add_load: i32,
}

impl Clone for ServerNode {
    // Constructor method to create a new ServerNode
    fn clone(&self) -> Self {
        ServerNode {
            ipaddr: self.ipaddr.clone(),
            gpus: self.gpus.clone(),
            stream: self.stream.clone(),
            virt_servers: self.virt_servers.clone(),
            metrics: self.metrics.clone(),
            add_load: self.add_load,
        }
    }
}

impl ServerNode {
    // Constructor method to create a new ServerNode
    pub fn new(
        ipaddr: String,
        gpus: Vec<Arc<RwLock<GPU>>>,
        stream: Arc<RwLock<StreamEnds<TcpStream>>>,
        virt_servers: Vec<Arc<RwLock<VirtServer>>>,
    ) -> Self {
        ServerNode {
            ipaddr,
            gpus,
            stream,
            virt_servers,
            metrics: None, // Default to None
            add_load: DEFAULT_ADD_LOAD, // Use default value
        }
    }

    // Method to reset add_load to default value
    pub fn reset_add_load(&mut self) {
        self.add_load = DEFAULT_ADD_LOAD;
    }

    pub fn get_default_load() -> i32 {
        return DEFAULT_ADD_LOAD;
    }
}

#[derive(Debug, Clone)]
pub struct VirtServer {
    pub ipaddr: String,
    pub compute_units: u32,
    pub memory: u64,
    pub rpc_id: u64,
    pub actual_units: u32,
    pub gpu: Arc<RwLock<GPU>>,
}



#[derive(Debug, Serialize, Deserialize)]
pub struct VMResources {
    pub vm_ip: String,
    pub host_ip: String,
    pub compute_units: u32,
    pub memory: u64,
}

pub struct VMResourcesGetter {
    mongo_collection: Option<Collection<VMResources>>,
}



impl VMResourcesGetter {

    pub fn new() -> Self {
        let config: Table = Utils::load_config_file(RMGR_CONFIG_PATH);

        let get_collection = || -> Option<Collection<VMResources>> {

            let db_details = config.get("vm-resource-db")?.as_table()?;
            let db_host = db_details.get("host")?.as_str()?;
            let db_port = db_details.get("port")?.as_integer()?;
            let db_user = db_details.get("user")?.as_str()?;
            let db_password = db_details.get("password")?.as_str()?;
            let db_dbname = db_details.get("dbname")?.as_str()?;

            let client_options = ClientOptions::builder()
            .hosts(vec![ServerAddress::Tcp {
                host: db_host.to_string(),
                port: Some(db_port as u16),
            }])
            .credential(Credential::builder()
                .username(db_user.to_string())
                .password(db_password.to_string())
                .build())
            .build();
            
            let client = Client::with_options(client_options).ok()?;

            Some(client.database(db_dbname).collection("vm_required_resources"))
        };

        Self {
            mongo_collection: get_collection(),
        }

    }

    pub fn get_vm_required_resources(&self, vm_ip: &String) -> Option<VMResources> {
        // let mut lock = self.mongo_client.try_lock().ok()?;
        let collection = self.mongo_collection.as_ref()?;
        let filter = doc! { "vm_ip": vm_ip };
        collection.find_one(filter, None).ok()?.and_then(|mut rsc| {
            rsc.memory = rsc.memory * 1024 * 1024;
            Some(rsc)
        })
    }

    pub fn set_vm_sm_resource( &self, vm_ip: &String, new_sm_core: i32,) -> Option<i32> {
        // Access the MongoDB collection
        if new_sm_core <= 0 {
            return None;
        }
        let collection = self.mongo_collection.as_ref()?;
        let filter = doc! { "vm_ip": vm_ip };
        let update = doc! { "$set": { "compute_units": new_sm_core } };

        // Set the options to return the old document
        let options = FindOneAndUpdateOptions::builder()
            .return_document(ReturnDocument::Before)
            .build();

        // Perform the update operation
        let result = collection
            .find_one_and_update(filter, update, options)
            .ok()?;

        // Extract the old sm_core value from the result
        result.map(|rsc| rsc.compute_units as i32)
    }


}

pub fn get_ckp_base_path() -> Option<String> {
    let config: Table = Utils::load_config_file(RMGR_CONFIG_PATH);
    config.get("migration")?.get("ckp-path")?.as_str().map(|s| s.to_string())
}

pub fn get_virt_server_deallocate_time() -> Option<u64> {

    if let Some(deallocate_time) = ConfigOptions::VIRT_SERVER_DEALLOCATE_TIME.read().unwrap().clone() {
        return deallocate_time;
    }

    let config: Table = Utils::load_config_file(RMGR_CONFIG_PATH);

    let enabled = config.get("virt-server-auto-deallocate")?.get("enabled")?.as_bool()?;
    if !enabled {
        return None;
    }
    let deallocate_time = config.get("virt-server-auto-deallocate")?.get("grace-period")?.as_integer()?;
    let deallocate_time = Some(deallocate_time as u64);

    ConfigOptions::VIRT_SERVER_DEALLOCATE_TIME.write().unwrap().replace(deallocate_time);
    deallocate_time
}

pub fn get_ports() -> (u16, u16) {
    let config: Table = Utils::load_config_file(RMGR_CONFIG_PATH);
    let node_port = config.get("ports").unwrap().get("node").unwrap().as_integer().unwrap() as u16;
    let client_port = config.get("ports").unwrap().get("client").unwrap().as_integer().unwrap() as u16;
    (node_port, client_port)
}

pub fn get_metrics_port() -> (u16, u16) {
    let config: Table = Utils::load_config_file(RMGR_CONFIG_PATH);
    let metric_port = config.get("metrics").unwrap().get("port").unwrap().as_integer().unwrap() as u16;
    let metric_interval = config.get("metrics").unwrap().get("interval").unwrap().as_integer().unwrap() as u16;
    (metric_port, metric_interval)
}
