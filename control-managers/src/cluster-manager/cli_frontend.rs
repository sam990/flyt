#![allow(dead_code)]

use std::{
    io::Write,
    os::unix::net::UnixStream,
};

use clap::{Parser, Subcommand};
use comfy_table::Table;
use common::{api_commands::FrontEndCommand, config::RMGR_CONFIG_PATH};

use crate::common::utils::StreamUtils;

#[path = "../common/mod.rs"]
mod common;


#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    ListVms,
    ListServernodes,
    ListVirtServers,
    ChangeConfig {
        #[arg(short, long, help = "IP address of the VM to change resources for")]
        ip: String,
        #[clap(flatten)]
        new_resources: NewResourcesOption,
    },
    Migrate {
        #[arg(short, long, help = "IP address of the VM to migrate")]
        ip: String,
        #[arg(short, long, help = "IP address of the server node to migrate to")]
        dstsnodeip: String,
        #[arg(short, long, help = "ID of the GPU to migrate to")]
        dstgpuid: u32,
        #[arg(short, long, help = "Number of SM cores to allocate")]
        sm_cores: u32,
        #[arg(short, long, help = "Amount of memory to allocate")]
        memory: u64,
    }
}
#[derive(Debug, clap::Args, Clone)]
#[group(required = true)]
struct NewResourcesOption {
    #[arg(short, long, help = "Number of SM cores to allocate")]
    sm_cores: Option<u32>,
    #[arg(short, long, help = "Amount of memory to allocate")]
    memory: Option<u64>,
}

pub fn get_stream_path() -> String {
    let config = common::utils::Utils::load_config_file(RMGR_CONFIG_PATH);
    config["ipc"]["frontend-socket"]
        .as_str()
        .unwrap()
        .to_string()
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let stream_path = get_stream_path();

    let stream = UnixStream::connect(stream_path).unwrap();

    match args.cmd {
        Commands::ListVms => {
            list_vms(stream);
        }
        Commands::ListServernodes => {
            list_servernodes(stream);
        }
        Commands::ListVirtServers => list_virt_servers(stream),
        Commands::ChangeConfig { ip, new_resources } => {
            change_resources(stream, ip, new_resources);
        }
        Commands::Migrate {
            ip,
            dstsnodeip,
            dstgpuid,
            sm_cores,
            memory,
        } => {
            migrate_vm(stream, ip, dstsnodeip, dstgpuid, sm_cores, memory);
        }
    }
}

fn migrate_vm(mut stream: UnixStream, ip: String, dstsnodeip: String, dstgpuid: u32, sm_cores: u32, memory: u64) {
    match stream.write_all(
        format!(
            "{}\n{},{},{},{},{}\n",
            FrontEndCommand::MIGRATE_VIRT_SERVER,
            ip,
            dstsnodeip,
            dstgpuid,
            sm_cores,
            memory
        )
        .as_bytes(),
    ) {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error writing to stream: {}", e);
            return;
        }
    }

    let mut reader = std::io::BufReader::new(stream);

    let response = match StreamUtils::read_response(&mut reader, 2) {
        Ok(response) => response,
        Err(e) => {
            log::error!("Error reading response: {}", e);
            return;
        }
    };

    println!("{}: {}", response[0], response[1]);
}


fn list_vms(mut stream: UnixStream) {
    match stream.write_all(format!("{}\n", FrontEndCommand::LIST_VMS).as_bytes()) {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error writing to stream: {}", e);
            return;
        }
    }
    
    let mut reader = std::io::BufReader::new(stream);

    let status = match StreamUtils::read_line(&mut reader) {
        Ok(status) => status,
        Err(e) => {
            log::error!("Error reading status: {}", e);
            return;
        }
    };

    if status != "200" {
        log::error!("Error: {}", status);
        let error_msg = match StreamUtils::read_line(&mut reader) {
            Ok(msg) => msg,
            Err(e) => {
                log::error!("Error reading error message: {}", e);
                return;
            }
        };
        log::error!("{}", error_msg);
        return;
    }

    let num_vms_str = match StreamUtils::read_line(&mut reader) {
        Ok(num_vms) => num_vms,
        Err(e) => {
            log::error!("Error reading number of VMs: {}", e);
            return;
        }
    };

    let num_vms = num_vms_str.parse::<usize>().unwrap();

    let mut table = Table::new();

    table.set_header(vec![
        "VM IP",
        "VirtServer IP",
        "VirtServer RPC ID",
        "SM Cores",
        "Memory",
        "Is Active",
    ]);

    for _ in 0..num_vms {
        let row = match StreamUtils::read_line(&mut reader) {
            Ok(row) => row,
            Err(e) => {
                log::error!("Error reading VM details: {}", e);
                return;
            }
        };
        let fields = row.split(',').collect::<Vec<&str>>();
        table.add_row(fields);
    }

    println!("{table}");
}

fn list_servernodes(mut stream: UnixStream) {
    match stream.write_all(format!("{}\n", FrontEndCommand::LIST_SERVER_NODES).as_bytes()) {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error writing to stream: {}", e);
            return;
        }
    }
    let mut reader = std::io::BufReader::new(stream);

    let status = match StreamUtils::read_line(&mut reader) {
        Ok(status) => status,
        Err(e) => {
            log::error!("Error reading status: {}", e);
            return;
        }
    };

    if status != "200" {
        log::error!("Error: {}", status);
        let error_msg = match StreamUtils::read_line(&mut reader) {
            Ok(msg) => msg,
            Err(e) => {
                log::error!("Error reading error message: {}", e);
                return;
            }
        };
        log::error!("{}", error_msg);
        return;
    }

    let num_servernodes_str = match StreamUtils::read_line(&mut reader) {
        Ok(num_servernodes) => num_servernodes,
        Err(e) => {
            log::error!("Error reading number of server nodes: {}", e);
            return;
        }
    };
    let num_servernodes =num_servernodes_str.parse::<usize>().unwrap();

    let mut response = String::new();

    // format: ipaddr,num_gpus
    for _ in 0..num_servernodes {
        let fields_str = match StreamUtils::read_line(&mut reader) {
            Ok(fields) => fields,
            Err(e) => {
                log::error!("Error reading server node details: {}", e);
                return;
            }
        };
        let fields = fields_str.split(',').collect::<Vec<&str>>();
        let ipaddr = fields[0];
        let num_gpus = fields[1].parse::<usize>().unwrap();

        response.push_str(format!("ServerNode IP: {}\n", ipaddr).as_str());
        let mut table = Table::new();

        // format: gpuid,name,memory,allocated_memory,compute_units,allocated_compute_units
        table.set_header(vec![
            "GPU ID",
            "GPU Name",
            "GPU Memory",
            "Allocated GPU Memory",
            "GPU Compute Units",
            "Allocated GPU Compute Units",
        ]);

        for _ in 0..num_gpus {
            let row_str = match StreamUtils::read_line(&mut reader) {
                Ok(row) => row,
                Err(e) => {
                    log::error!("Error reading GPU details: {}", e);
                    return;
                }
            };
            let fields = row_str.split(',').collect::<Vec<&str>>();
            table.add_row(fields);
        }

        response.push_str(&format!("{table}\n"));
    }

    println!("{}", response);
}

fn list_virt_servers(mut stream: UnixStream) {
    match stream.write_all(format!("{}\n", FrontEndCommand::LIST_VIRT_SERVERS).as_bytes()) {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error writing to stream: {}", e);
            return;
        }
    }

    let mut reader = std::io::BufReader::new(stream);

    let status = match StreamUtils::read_line(&mut reader) {
        Ok(status) => status,
        Err(e) => {
            log::error!("Error reading status: {}", e);
            return;
        }
    };

    if status != "200" {
        log::error!("Error: {}", status);
        let error_msg = match StreamUtils::read_line(&mut reader) {
            Ok(msg) => msg,
            Err(e) => {
                log::error!("Error reading error message: {}", e);
                return;
            }
        };
        log::error!("{}", error_msg);
        return;
    }

    let num_virtual_servers_str = match StreamUtils::read_line(&mut reader) {
        Ok(num_virt_servers) => num_virt_servers,
        Err(e) => {
            log::error!("Error reading number of virtual servers: {}", e);
            return;
        }
    };
    let num_virt_servers = num_virtual_servers_str.parse::<usize>().unwrap();

    // format: ipaddr,rpc_id,gpu_id,compute_units,memory
    let mut table = Table::new();
    table.set_header(vec![
        "VirtServer IP",
        "VirtServer RPC ID",
        "GPU ID",
        "Compute Units",
        "Memory",
    ]);

    for _ in 0..num_virt_servers {
        let fields_str = match StreamUtils::read_line(&mut reader) {
            Ok(fields) => fields,
            Err(e) => {
                log::error!("Error reading virtual server details: {}", e);
                return;
            }
        };
        let fields = fields_str.split(',').collect::<Vec<&str>>();
        table.add_row(fields);
    }

    println!("{}", table);
}

fn change_resources(mut stream: UnixStream, vm_ip: String, new_resources: NewResourcesOption) {
    let command = if new_resources.sm_cores.is_some() && new_resources.memory.is_some() {
        format!(
            "{}\n{},{},{}\n",
            FrontEndCommand::CHANGE_SM_CORES_AND_MEMORY,
            vm_ip,
            new_resources.sm_cores.unwrap(),
            new_resources.memory.unwrap()
        )
    } else if let Some(sm_cores) = new_resources.sm_cores {
        format!(
            "{}\n{},{}\n",
            FrontEndCommand::CHANGE_SM_CORES,
            vm_ip,
            sm_cores
        )
    } else if let Some(memory) = new_resources.memory {
        format!("{}\n{},{}\n", FrontEndCommand::CHANGE_MEMORY, vm_ip, memory)
    } else {
        "".to_string()
    };

    if command.is_empty() {
        log::error!("Invalid command");
        return;
    }

    match stream.write_all(command.as_bytes()) {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error writing to stream: {}", e);
            return;
        }
    }

    let mut reader = std::io::BufReader::new(stream);

    let response = match StreamUtils::read_response(&mut reader, 2) {
        Ok(response) => response,
        Err(e) => {
            log::error!("Error reading response: {}", e);
            return;
        }
    };

    println!("{}: {}", response[0], response[1]);
}
