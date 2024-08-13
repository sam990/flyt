const { MongoClient } = require('mongodb');

const config = {
    host: "localhost",
    port: 27017,
    user: "adminUser",
    password: "flyt",
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

	// Define the filter and the update operation
        const filter = { vm_ip: "10.129.28.180" }; // Criteria to find the document
	var doc = {
	    vm_ip: "10.129.28.180",
	    host_ip: "10.129.2.27",
	    compute_units: 64,
	    memory: 8196
	};
        const update = {
            $set: doc
        };

        // Perform the upsert operation
        const result = await collection.updateOne(filter, update, { upsert: true });
        console.log(`Matched ${result.matchedCount} document(s) and modified ${result.modifiedCount} document(s).`);

        if (result.upsertedCount > 0) {
            console.log("Inserted a new document with _id:", result.upsertedId._id);
        } else {
            console.log("Updated existing document.");
        }
    } catch (err) {
        console.error("Error connecting to the database or performing operations:", err);
    } finally {
        await client.close();
        console.log("Closed the database connection");
    }
}

main().catch(console.error);

