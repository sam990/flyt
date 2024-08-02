use mongodb::{bson::doc, options::ClientOptions, Client, Collection};
use serde::{Deserialize, Serialize};
use tokio;

#[derive(Debug, Serialize, Deserialize)]
struct VMResources {
    vm_ip: String,
    host_ip: String,
    compute_units: i32,
    memory: i64, // Assuming memory is in GB
}

struct Config {
    host: String,
    port: u16,
    user: String,
    password: String,
    dbname: String,
}

impl Config {
    fn new() -> Self {
        // Replace these with actual configuration values
        Self {
            host: "localhost".to_string(),
            port: 27017,
            user: "adminUser".to_string(),
            password: "flyt".to_string(),
            dbname: "flyt".to_string(),
        }
    }
}

struct MyService {
    collection: Collection<VMResources>,
}

impl MyService {
    async fn new() -> Self {
        let config = Config::new();

        let client_uri = format!(
            "mongodb://{}:{}@{}:{}",
            config.user, config.password, config.host, config.port
        );
        let client_options = ClientOptions::parse(&client_uri).await.unwrap();
        let client = Client::with_options(client_options).unwrap();
        let database = client.database(&config.dbname);
        let collection = database.collection::<VMResources>("vm_required_resources");

        MyService { collection }
    }

    async fn insert_vm_resource(&self, resource: VMResources) {
        self.collection.insert_one(resource, None).await.unwrap();
    }

    async fn get_vm_required_resources(&self, vm_ip: &str) -> Option<VMResources> {
        let filter = doc! { "vm_ip": vm_ip };
        self.collection.find_one(filter, None).await.unwrap().map(|mut rsc| {
            rsc.memory *= 1024 * 1024; // Convert memory from GB to MB if needed
            rsc
        })
    }
}

#[tokio::main]
async fn main() {
    let service = MyService::new().await;

    let new_resource = VMResources {
        vm_ip: "192.168.1.100".to_string(),
        host_ip: "192.168.1.1".to_string(),
        compute_units: 4,
        memory: 16, // Assuming this is in GB
    };

    // Insert a document
    //service.insert_vm_resource(new_resource).await;
    //println!("Inserted document");

    // Retrieve a document
    if let Some(vm_resource) = service.get_vm_required_resources("10.129.26.124").await {
        println!("Retrieved document: {:?}", vm_resource);
    } else {
        println!("Document not found");
    }
}

