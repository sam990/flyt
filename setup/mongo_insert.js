db = connect("mongodb://localhost:27017/flyt");

const existingDoc = db.vm_required_resources.findOne({ vm_ip: "10.129.26.124" });
if (existingDoc) {
    db.vm_required_resources.updateOne(
        { vm_ip: "10.129.26.124" },
        {
            $set: {
                host_ip: "10.129.27.234",
                compute_units: 10,
                memory: 250
            }
        }
    );
    print("updated document into vm_required_resources");
} else {
    db.vm_required_resources.insertOne({
        vm_ip: "10.129.26.124",
        host_ip: "10.129.27.234",
        compute_units: 10,
        memory: 250
    });
    print("Inserted document into vm_required_resources");
}


/*
// Authentication and connection details
const username = "adminUser";
const password = "flyt";
const database = "flyt";
const host = "localhost:27017";

// Connection URI
const uri = `mongodb://${username}:${password}@${host}/${database}`;

// Use the Mongo shell to connect
conn = new Mongo(uri);
db = conn.getDB(database);

// Define the collection
var collection = db.vm_required_resources;

// The document to insert or update
var doc = {
    vm_ip: "10.129.26.124",
    host_ip: "10.129.27.234",
    compute_units: 64,
    memory: 250
};

// Check if the document exists
var existingDoc = collection.findOne({ vm_ip: doc.vm_ip });

if (existingDoc) {
    // Update the existing document
    collection.updateOne(
        { vm_ip: doc.vm_ip },
        { $set: doc }
    );
    print("Document updated.");
} else {
    // Insert the new document
    collection.insertOne(doc);
    print("Document inserted.");
}
*/
