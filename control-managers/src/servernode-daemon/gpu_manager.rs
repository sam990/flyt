use nvml_wrapper::{enum_wrappers::device::Clock, Nvml};
use std::{path::Path, process::Command};

const GPU_CORES_PROGRAM: &str = "get-gpu-sm-cores";

#[derive(Debug,Clone)]
pub struct GPU {
    pub name: String,
    pub memory: u64,
    pub sm_cores: u32,
    pub total_cores: u32,
    pub max_clock: u32,
    pub gpu_id: u32,
}

pub fn get_all_gpus() -> Option<Vec<GPU>> {
    
    if Path::new(GPU_CORES_PROGRAM).exists() {
        println!("{} not found", GPU_CORES_PROGRAM);
        return None;
    }

    let nvml = Nvml::init().ok()?;
    let num_devices = nvml.device_count().ok()?;

    let mut gpus = Vec::new();

    for i in 0..num_devices {
        let device = nvml.device_by_index(i).ok()?;
        let name = device.name().ok()?;
        let memory = device.memory_info().ok()?.free;
        let max_clock = device.max_clock_info(Clock::SM).ok()?;
        let sm_cores = Command::new(GPU_CORES_PROGRAM).arg(i.to_string()).output().ok()?;
        let total_cores = device.num_cores().ok()?;

        if !sm_cores.status.success() {
            println!("Error getting SM cores for GPU {}", i);
            continue;
        } 

        let gpu_id = i;
        let sm_cores = String::from_utf8(sm_cores.stdout).ok()?.trim().parse().ok()?;

        gpus.push(GPU {
            name: name,
            memory: memory,
            sm_cores: sm_cores,
            total_cores: total_cores,
            max_clock: max_clock,
            gpu_id: gpu_id
        });
    }

    Some(gpus)
}