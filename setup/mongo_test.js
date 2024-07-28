const { MongoClient } = require('mongodb');

const config = {
    host: "localhost",
    port: 27017,
    user: "adminUser",
    password: "securePassword",
    dbname: "flyt"
};

async function main() {
    const uri = `mongodb://${config.user}:${config.password}@${config.host}:${config.port}/${config.dbname}`;
    const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

    try {
        await client.connect();
        console.log("Connected to the database");

        const database = client.db(config.dbname);
        const collection = database.collection("vm_required_resources");

        // Find a document
        const vm_ip = "10.129.26.124";
        const document = await collection.findOne({ vm_ip: vm_ip });
        if (document) {
            document.memory *= 1024 * 1024; // Convert memory from GB to MB if needed
            console.log("Retrieved document:", document);
        } else {
            console.log("Document not found");
        }
    } catch (err) {
        console.error("Error connecting to the database or performing operations:", err);
    } finally {
        await client.close();
        console.log("Closed the database connection");
    }
}

main().catch(console.error);

