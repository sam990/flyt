use std::sync::RwLock;
use std::sync::Arc;
use std::net::TcpStream;
use toml::Table;
use mongodb::{options::{ClientOptions, ServerAddress, Credential}, sync::Client, sync::Collection, bson::doc};
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
        }
    }
}

#[derive(Debug)]
pub struct ServerNode {
    pub ipaddr: String,
    pub gpus: Vec<Arc<RwLock<GPU>>>,
    pub stream: Arc<RwLock<StreamEnds<TcpStream>>>,
    pub virt_servers: Vec<Arc<RwLock<VirtServer>>>,
}

impl Clone for ServerNode {
    fn clone(&self) -> Self {
        ServerNode {
            ipaddr: self.ipaddr.clone(),
            gpus: self.gpus.clone(),
            stream: self.stream.clone(),
            virt_servers: self.virt_servers.clone()
        }
    }
}

#[derive(Debug, Clone)]
pub struct VirtServer {
    pub ipaddr: String,
    pub compute_units: u32,
    pub memory: u64,
    pub rpc_id: u64,
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
        collection.find_one(filter, None).ok()?
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