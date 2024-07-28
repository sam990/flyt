use mongodb::{options::{ClientOptions, ServerAddress, Credential}, Client, Collection, bson::{self, doc, Document, to_document}, error::Result};
use serde::{Deserialize, Serialize};
use anyhow::Result as AnyResult; // Importing Result alias from anyhow crate

struct Config {
    host: String,
    port: u16,
    user: String,
    password: String,
    dbname: String,
}

impl Config {
    fn new() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 27017,
            user: "adminUser".to_string(),
            password: "securePassword".to_string(),
            dbname: "flyt".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VMResources {
    pub vm_ip: String,
    pub host_ip: String,
    pub compute_units: u32,
    pub memory: u64,
}

pub struct VMResourcesGetter {
    mongo_collection: Option<Collection<Document>>,
}

impl VMResourcesGetter {
    pub fn new() -> Self {
        let config = Config::new();

        let get_collection = || -> Option<Collection<Document>> {
            let db_host = config.host;
            let db_port = config.port;
            let db_user = config.user;
            let db_password = config.password;
            let db_dbname = config.dbname;

            let client_options = ClientOptions::builder()
                .hosts(vec![ServerAddress::Tcp {
                    host: db_host.clone(),
                    port: Some(db_port),
                }])
                .credential(Credential::builder()
                    .username(db_user.clone())
                    .password(db_password.clone())
                    .build())
                .build();

            let client = Client::with_options(client_options).ok()?;
            let collection_name = "vm_required_resources";
            Some(client.database(&db_dbname).collection(collection_name))
        };

        Self {
            mongo_collection: get_collection(),
        }
    }

    pub async fn get_vm_required_resources(&self, vm_ip: &String) -> Option<VMResources> {
        let collection = self.mongo_collection.as_ref()?;
        let filter = doc! { "vm_ip": vm_ip };
        if let Ok(document) = collection.find_one(filter, None).await {
            document.and_then(|doc| bson::from_document(doc).ok())
        } else {
            None
        }
    }

    pub async fn insert_vm_required_resources(&self, vm_ip: &String, host_ip: &String, compute_units: u32, memory: u64) -> AnyResult<()> {
        let collection = self.mongo_collection.as_ref().ok_or_else(|| anyhow::anyhow!("Mongo collection not available"))?;

        let document = VMResources {
            vm_ip: vm_ip.clone(),
            host_ip: host_ip.clone(),
            compute_units,
            memory,
        };

        let bson_document = to_document(&document)?; // Convert VMResources to BSON Document
        let result = collection.insert_one(bson_document, None).await?;
        println!("Inserted document: {:?}", result.inserted_id);

        Ok(())
    }
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    let vm_getter = VMResourcesGetter::new();
    let vm_ip_to_find = "10.129.26.124".to_string();
    let vm_ip_to_insert = "10.129.26.124".to_string();
    let host_ip_to_insert = "10.129.27.234".to_string();

    if let Some(vm_resources) = vm_getter.get_vm_required_resources(&vm_ip_to_find).await {
        println!("Retrieved VM resources: {:?}", vm_resources);
    } else {
        println!("No VM resources found for IP: {}", vm_ip_to_find);
    }

    vm_getter.insert_vm_required_resources(&vm_ip_to_insert, &host_ip_to_insert, 10, 8149).await?;

    if let Some(vm_resources) = vm_getter.get_vm_required_resources(&vm_ip_to_insert).await {
        println!("Retrieved VM resources: {:?}", vm_resources);
    } else {
        println!("No VM resources found for IP: {}", vm_ip_to_find);
    }

    Ok(())
}

