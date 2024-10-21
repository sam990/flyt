#![allow(dead_code)]

use std::{io::{Read, Write}, net::TcpListener, os::unix::net::UnixStream, time::SystemTime};

use cli_frontend::{change_resources, get_stream_path, increase_resources, migrate_vm_auto, NewResourcesOption};

mod cli_frontend;
#[path = "../common/mod.rs"]
mod common;

const INCREASE_SM_DELTA: u32 = 8;

fn get_stream() -> UnixStream {
    let stream_path = get_stream_path();
    UnixStream::connect(stream_path).unwrap()
}

fn main() {
    let mut curr_sm_cores = 16;
    let mut migration_done = false;
    let migration_vm_mem = 8192 * 1024 * 1024;

    let listener = TcpListener::bind("0.0.0.0:32578").unwrap();
    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        println!("Connection established!");
        let vm_ip = stream.peer_addr().unwrap().ip().to_string();

        let mut buf = [0; 1];

        stream.read(&mut buf).unwrap();


        match buf[0] {
            1u8 => {
                println!("Received inc_cores command");             

                let mgr_stream = get_stream();

                let res = increase_resources(mgr_stream, &vm_ip, NewResourcesOption{ sm_cores: Some(INCREASE_SM_DELTA), memory: None });

                if res.success {
                    stream.write(&[0u8]).unwrap();
                } else {
                    stream.write(&[1u8]).unwrap();
                }

                let mut file = match std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open("resource_tracker.log")
                {
                    Ok(file) => file,
                    Err(e) => {
                        log::error!("Error opening log file: {}", e);
                        return;
                    }
                };

                let current_time = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_micros();
                if let Err(e) = writeln!(file, "{}: {},{},{},{},{},{}", current_time, vm_ip, res.success, res.sm_cores, res.memory, res.time_taken.as_micros(), res.error) {
                    log::error!("Error writing to log file: {}", e);
                }

            }
            
            2u8 => {
                println!("Received inc_cores_with_migration command");
                if migration_done && curr_sm_cores >= 64 {
                    println!("Cannot increment cores. Already at max capacity.");
                    stream.write(&[1u8]).unwrap();
                } else if !migration_done && curr_sm_cores >= 48 {
                    curr_sm_cores = 64;
                    let mgr_stream = get_stream();

                    // migrate_vm(mgr_stream, vm_ip, migration_vm_ip.to_string(), 0, sm_cores, memory)
                    migrate_vm_auto(mgr_stream, vm_ip, curr_sm_cores, migration_vm_mem);
                    migration_done = true;
                    stream.write(&[0u8]).unwrap();
                }
                else {
                    println!("Incrementing cores...");
                    curr_sm_cores += 16;
                    let mgr_stream = get_stream();

                    change_resources(mgr_stream, &vm_ip, NewResourcesOption{ sm_cores: Some(curr_sm_cores), memory: None });

                    stream.write(&[0u8]).unwrap();
                }
            }

            _ => {
                println!("Received unknown command: {}", buf[0]);
            }
        }

    }

}