#[non_exhaustive]
pub struct FlytApiCommand;


impl FlytApiCommand {
    pub const CLIENTD_VCUDA_PAUSE: &'static str = "CLIENTD_VCUDA_PAUSE";
    pub const CLIENTD_VCUDA_CHANGE_VIRT_SERVER: &'static str = "CLIENTD_VCUDA_CHANGE_VIRT_SERVER";
    pub const CLIENTD_VCUDA_RESUME: &'static str = "CLIENTD_VCUDA_RESUME";
    pub const PING: &'static str = "PING";
    pub const CLIENTD_RMGR_CONNECT: &'static str = "CLIENTD_RMGR_CONNECT";
    pub const RMGR_CLIENTD_PAUSE: &'static str = "RMGR_CLIENTD_PAUSE";
    pub const RMGR_CLIENTD_RESUME: &'static str = "RMGR_CLIENTD_RESUME";
    pub const RMGR_CLIENTD_CHANGE_VIRT_SERVER: &'static str = "RMGR_CLIENTD_CHANGE_VIRT_SERVER";
    pub const CLIENTD_RMGR_ZERO_VCUDA_CLIENTS: &'static str = "CLIENTD_RMGR_ZERO_VCUDA_CLIENTS";
    pub const RMGR_CLIENTD_DEALLOC_VIRT_SERVER: &'static str = "RMGR_CLIENTD_DEALLOC_VIRT_SERVER";
    pub const RMGR_SNODE_DEALLOC_VIRT_SERVER: &'static str = "RMGR_SNODE_DEALLOC_VIRT_SERVER";
    pub const RMGR_SNODE_SEND_GPU_INFO: &'static str = "RMGR_SNODE_SEND_GPU_INFO";
    pub const RMGR_SNODE_ALLOC_VIRT_SERVER: &'static str = "RMGR_SNODE_ALLOC_VIRT_SERVER";
    pub const RMGR_SNODE_CHANGE_RESOURCES: &'static str = "RMGR_SNODE_CHANGE_RESOURCES";
    pub const SNODE_VIRTS_CHANGE_RESOURCES: &'static str = "SNODE_VIRTS_CHANGE_RESOURCES";
    pub const SNODE_VIRTS_CHECKPOINT: &'static str = "SNODE_VIRTS_CHECKPOINT";
    pub const SNODE_VIRTS_RESTORE: &'static str = "SNODE_VIRTS_RESTORE";
    pub const RMGR_SNODE_CHECKPOINT: &'static str = "RMGR_SNODE_CHECKPOINT";
    pub const RMGR_SNODE_ALLOC_RESTORE: &'static str = "RMGR_SNODE_ALLOC_RESTORE";
}

pub struct FrontEndCommand;

impl FrontEndCommand {
    pub const LIST_VMS: &'static str = "LIST_VMS";
    pub const LIST_SERVER_NODES: &'static str = "LIST_SERVER_NODES";
    pub const LIST_VIRT_SERVERS: &'static str = "LIST_VIRT_SERVERS";
    pub const CHANGE_SM_CORES: &'static str = "CHANGE_SM_CORES";
    pub const CHANGE_MEMORY: &'static str = "CHANGE_MEMORY";
    pub const CHANGE_SM_CORES_AND_MEMORY: &'static str = "CHANGE_SM_CORES_AND_MEMORY";
    pub const MIGRATE_VIRT_SERVER: &'static str = "MIGRATE_VIRT_SERVER";
}