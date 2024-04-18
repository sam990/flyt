
fn main() {
    cc::Build::new()
        .cuda(true)
        .file("src/servernode-daemon/gpu_cores_getter.c")
        .compile("gpu_cores_getter")
}