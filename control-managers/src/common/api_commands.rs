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
}