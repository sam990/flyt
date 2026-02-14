// Copyright (c) 2024-2026 SynerG Lab, IITB

#![allow(dead_code)]

use std::{
    io::Write, os::unix::net::UnixStream
};

use clap::{Parser, Subcommand};
use comfy_table::Table;
use common::{api_commands::FrontEndCommand, config::RMGR_CONFIG_PATH, types::IncResourcesResult};

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
        #[arg(short, short, long, help = "IP address of the VM to change resources for")]
        ip: String,
        #[arg(short, short, long, help = "Client application ID to change resources for")]
        client_id: i32,
        #[clap(flatten)]
        new_resources: NewResourcesOption,
    },
    IncResources {
        #[arg(short, short, long, help = "IP address of the VM to change resources for")]
        ip: String,
        #[arg(short, short, long, help = "Client application ID to change resources for")]
        client_id: i32,
        #[clap(flatten)]
        new_resources: NewResourcesOption,
    },
    DecResources {
        #[arg(short, short, long, help = "IP address of the VM to change resources for")]
        ip: String,
        #[arg(short, short, long, help = "Client application ID to change resources for")]
        client_id: i32,
        #[clap(flatten)]
        new_resources: NewResourcesOption,
    },
    Migrate {
        #[arg(short, short, long, help = "IP address of the VM to migrate")]
        ip: String,
        #[arg(short, short, short, long, help = "Client application ID to change resources for")]
        client_id: i32,
        #[arg(short, short, long, help = "IP address of the server node to migrate to")]
        dstsnodeip: String,
        #[arg(short, short, long, help = "ID of the GPU to migrate to")]
        dstgpuid: u32,
        #[arg(short, short, long, help = "Number of SM cores to allocate")]
        sm_cores: u32,
        #[arg(short, short, long, help = "Amount of memory to allocate (MB)")]
        memory: u64,
    },
    MigrateAuto {
        #[arg(short, short, long, help = "IP address of the VM to migrate")]
        ip: String,
        #[arg(short, short, long, help = "Client application ID to change resources for")]
        client_id: i32,
        #[arg(short, short, long, help = "Number of SM cores to allocate")]
        sm_cores: u32,
        #[arg(short, short, long, help = "Amount of memory to allocate (MB)")]
        memory: u64,
    },
    VMGrouping {
        #[arg(short, short, long, help = "IP address of the VM to migrate")]
        ip: String,
        #[arg(short, short, long, help = "Client application ID to change resources for")]
        enable: i32,
    }
}
#[derive(Debug, clap::Args, Clone)]
#[group(required = true)]
pub struct NewResourcesOption {
    #[arg(short, long, help = "Number of SM cores to allocate")]
    pub sm_cores: Option<u32>,
    #[arg(short, long, help = "Amount of memory to allocate (MB)")]
    pub memory: Option<u64>,
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

    let stream_org = UnixStream::connect(stream_path);
    let stream = match stream_org {
        Ok(stream) => stream,
        Err(e) => {
            log::error!("Error stream: {}", e);
            return ;
        }
    };

    match args.cmd {
        Commands::ListVms => {
            list_vms(stream);
        }
        Commands::ListServernodes => {
            list_servernodes(stream);
        }
        Commands::ListVirtServers => list_virt_servers(stream),
        Commands::ChangeConfig { ip, client_id, mut new_resources } => {
            new_resources.memory = new_resources.memory.map(|x| x * 1024 * 1024);
            change_resources(stream, &ip, client_id, new_resources);
        }
        Commands::Migrate {
            ip,
            client_id,
            dstsnodeip,
            dstgpuid,
            sm_cores,
            memory,
        } => {
            let mem_bytes = memory * 1024 * 1024;
            migrate_vm(stream, ip, client_id, dstsnodeip, dstgpuid, sm_cores, mem_bytes);
        },
        Commands::MigrateAuto {
            ip,
            client_id,
            sm_cores,
            memory,
        } => {
            let mem_bytes = memory * 1024 * 1024;
            migrate_vm_auto(stream, ip, client_id, sm_cores, mem_bytes);
        },
        Commands::IncResources { ip, client_id, new_resources } => {
            increase_resources(stream, &ip, client_id, new_resources, false);
        },
        Commands::DecResources { ip, client_id, new_resources } => {
            increase_resources(stream, &ip, client_id, new_resources, true);
        },
        Commands::VMGrouping { ip, enable } => {
            set_vm_grouping(stream, &ip, enable);
        },
    }
}

pub fn migrate_vm(mut stream: UnixStream, ip: String, client_id: i32, dstsnodeip: String, dstgpuid: u32, sm_cores: u32, memory: u64) {

    let time_begin = std::time::Instant::now();

    match stream.write_all(
        format!(
            "{}\n{},{},{},{},{},{}\n",
            FrontEndCommand::MIGRATE_VIRT_SERVER,
            ip,
            client_id,
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

    let time_end = std::time::Instant::now();

    println!("{}: {}", response[0], response[1]);
    println!("Time taken: {:?}", time_end - time_begin);

}

pub fn migrate_vm_auto(mut stream: UnixStream, ip: String, client_id: i32, sm_cores: u32, memory: u64) {

    let time_begin = std::time::Instant::now();

    match stream.write_all(
        format!(
            "{}\n{},{},{},{}\n",
            FrontEndCommand::MIGRATE_VIRT_SERVER_AUTO,
            ip,
            client_id,
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

    let time_end = std::time::Instant::now();

    println!("{}: {}", response[0], response[1]);
    println!("Time taken: {:?}", time_end - time_begin);

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
        "CLIENT ID",
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

pub fn change_resources(mut stream: UnixStream, vm_ip: &String, client_id: i32, new_resources: NewResourcesOption) {
    let command = if new_resources.sm_cores.is_some() && new_resources.memory.is_some() {
        format!(
            "{}\n{},{},{},{}\n",
            FrontEndCommand::CHANGE_SM_CORES_AND_MEMORY,
            vm_ip,
            client_id,
            new_resources.sm_cores.unwrap(),
            new_resources.memory.unwrap()
        )
    } else if let Some(sm_cores) = new_resources.sm_cores {
        format!(
            "{}\n{},{},{}\n",
            FrontEndCommand::CHANGE_SM_CORES,
            vm_ip,
            client_id,
            sm_cores
        )
    } else if let Some(memory) = new_resources.memory {
        format!("{}\n{},{},{}\n", FrontEndCommand::CHANGE_MEMORY, vm_ip, client_id, memory)
    } else {
        "".to_string()
    };

    if command.is_empty() {
        log::error!("Invalid command");
        return;
    }

    let time_begin = std::time::Instant::now();

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

    let time_end = std::time::Instant::now();

    println!("{}: {}", response[0], response[1]);
    println!("Time taken: {:?}", time_end - time_begin);
}


pub fn increase_resources(mut stream: UnixStream, vm_ip: &String, client_id: i32, new_resources: NewResourcesOption, decrease: bool) -> IncResourcesResult {
    let time_begin = std::time::Instant::now();

    let sm_inc = new_resources.sm_cores.unwrap_or(0);
    let mem_inc = new_resources.memory.unwrap_or(0);
    let mut cmd = FrontEndCommand::INCREASE_RESOURCES;

    if decrease == true {
        cmd = FrontEndCommand::DECREASE_RESOURCES;
    }

    match stream.write_all(
        format!(
            "{}\n{},{},{},{}\n",
            cmd,
            vm_ip,
            client_id,
            sm_inc,
            mem_inc
        )
        .as_bytes(),
    ) {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error writing to stream: {}", e);
            return IncResourcesResult{
                error: format!("Error writing to stream: {}", e),
                ..Default::default()
            };
        }
    }

    let mut reader = std::io::BufReader::new(stream);

    let response = match StreamUtils::read_response(&mut reader, 2) {
        Ok(response) => response,
        Err(e) => {
            log::error!("Error reading response: {}", e);
            return IncResourcesResult{
                error: format!("Error reading response: {}", e),
                ..Default::default()
            };
        }
    };

    let time_end = std::time::Instant::now();

    println!("Time taken: {:?}", time_end - time_begin);

    let time_taken = time_end - time_begin;

    println!("{}: {}", response[0], response[1]);

    let response_parts = response[1].split(',').collect::<Vec<&str>>();

    if response_parts.len() != 3 {
        log::error!("Invalid response: {}", response[1]);
        return IncResourcesResult{
            error: format!("Invalid response: {}", response[1]),
            ..Default::default()
        };
    }

    let sm_cores = response_parts[0].parse::<u32>().unwrap();
    let memory = response_parts[1].parse::<u64>().unwrap();
    let description = response_parts[2].to_string();

    
    return IncResourcesResult{
        success: response[0] == "200",
        sm_cores,
        memory,
        error: description,
        time_taken: time_taken
    };

}

pub fn set_vm_grouping(mut stream: UnixStream, vm_ip: &String, enable: i32) {
    let time_begin = std::time::Instant::now();

    let mut cmd = FrontEndCommand::DISABLE_VM_GROUPING;

    if enable == 1 {
        cmd = FrontEndCommand::ENABLE_VM_GROUPING;
    }

    match stream.write_all(
        format!(
            "{}\n{}\n",
            cmd,
            vm_ip
        )
        .as_bytes(),
    ) {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error writing to stream: {}", e);
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
    println!("{}", status);
}
