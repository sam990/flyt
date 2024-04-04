use std::sync::RwLock;
use std::{fs::File, io::Read, sync::Arc};
use std::net::TcpStream;
use toml::Table;
use mongodb::{options::{ClientOptions, ServerAddress, Credential}, sync::Client, sync::Collection, bson::doc};
use serde::{Deserialize, Serialize};

const RESOURCE_MANAGER_CONFIG: &str = "/home/sam/Projects/flyt/control-managers/resource-mgr-config.toml";
struct ConfigOptions;

impl ConfigOptions {
    const VIRT_SERVER_DEALLOCATE_TIME: RwLock<Option<Option<u64>>> = RwLock::new(None);
}

#[derive(Debug, Clone)]
pub struct GPU {
    pub name: String,
    pub memory: u64,
    pub compute_units: u64,
    pub compute_power: u64,
    pub gpu_id: u64,
    pub allocated_memory: u64,
    pub allocated_compute_units: u64,
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
    pub stream: TcpStream,
    pub virt_servers: Vec<VirtServer>,
}

impl Clone for ServerNode {
    fn clone(&self) -> Self {
        ServerNode {
            ipaddr: self.ipaddr.clone(),
            gpus: self.gpus.clone(),
            stream: self.stream.try_clone().unwrap(),
            virt_servers: self.virt_servers.clone()
        }
    }
}

#[derive(Debug, Clone)]
pub struct VirtServer {
    pub ipaddr: String,
    pub compute_units: u64,
    pub memory: u64,
    pub rpc_id: u16,
    pub gpu: Arc<RwLock<GPU>>,
}



#[derive(Debug, Serialize, Deserialize)]
pub struct VMResources {
    pub vm_ip: String,
    pub host_ip: String,
    pub compute_units: u64,
    pub memory: u64,
}

pub struct VMResourcesGetter {
    mongo_collection: Option<Collection<VMResources>>,
}



impl VMResourcesGetter {

    pub fn new() -> Self {
        let mut config_file = File::open(RESOURCE_MANAGER_CONFIG).unwrap();
        let mut config_str = String::new();
        config_file.read_to_string(&mut config_str).unwrap();

        let config: Table = config_str.parse::<Table>().unwrap();

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

        VMResourcesGetter {
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



pub fn get_virt_server_deallocate_time() -> Option<u64> {

    if let Some(deallocate_time) = ConfigOptions::VIRT_SERVER_DEALLOCATE_TIME.read().unwrap().clone() {
        return deallocate_time;
    }


    let mut config_file = File::open(RESOURCE_MANAGER_CONFIG).unwrap();
    let mut config_str = String::new();
    config_file.read_to_string(&mut config_str).unwrap();

    let config: Table = config_str.parse::<Table>().unwrap();

    let enabled = config.get("virt-server-auto-deallocate")?.get("enabled")?.as_bool()?;
    if !enabled {
        return None;
    }
    let deallocate_time = config.get("virt-server-auto-deallocate")?.get("grace-period")?.as_integer()?;
    let deallocate_time = Some(deallocate_time as u64);

    ConfigOptions::VIRT_SERVER_DEALLOCATE_TIME.write().unwrap().replace(deallocate_time);
    deallocate_time
}