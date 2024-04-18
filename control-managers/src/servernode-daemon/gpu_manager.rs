use nvml_wrapper::{enum_wrappers::device::Clock, Nvml};

extern "C" {
    fn get_gpu_cores(device_id: u32) -> i32;
}

#[derive(Debug,Clone)]
pub struct GPU {
    pub name: String,
    pub memory: u64,
    pub sm_cores: u32,
    pub total_cores: u32,
    pub max_clock: u32,
    pub gpu_id: u32,
    pub virt_servers: Vec<u32>
}

pub fn get_all_gpus() -> Option<Vec<GPU>> {
    


    let nvml = Nvml::init().ok()?;
    let num_devices = nvml.device_count().ok()?;

    let mut gpus = Vec::new();

    for i in 0..num_devices {
        let device = nvml.device_by_index(i).ok()?;
        let name = device.name().ok()?;
        let memory = device.memory_info().ok()?.free;
        let max_clock = device.max_clock_info(Clock::SM).ok()?;
        let sm_cores = unsafe { get_gpu_cores(i) };
        let total_cores = device.num_cores().ok()?;

        if sm_cores == -1 {
            println!("Error getting SM cores for GPU {}", i);
            continue;
        } 

        let gpu_id = i;

        gpus.push(GPU {
            name: name,
            memory: memory,
            sm_cores: sm_cores as u32,
            total_cores: total_cores,
            max_clock: max_clock,
            gpu_id: gpu_id,
            virt_servers: Vec::new()
        });
    }

    Some(gpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_all_gpus() {
        let gpus = get_all_gpus();
        assert!(gpus.is_some());
        println!("{:?}", gpus.unwrap());
    }
}