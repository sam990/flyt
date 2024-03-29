#[non_exhaustive]
pub struct FlytApiCommand;


impl FlytApiCommand {
    pub const PAUSE_VCUDA_CLIENTS: &'static str = "PAUSE_VCUDA_CLIENTS";
    pub const CHANGE_VCUDA_VIRT_SERVER: &'static str = "CHANGE_VCUDA_VIRT_SERVER";
    pub const RESUME_VCUDA_CLIENTS: &'static str = "RESUME_VCUDE_CLIENTS";
    pub const PING: &'static str = "PING";
    pub const CLIENTD_RMGR_CONNECT: &'static str = "CLIENTD_RMGR_CONNECT";
}