db = connect("mongodb://localhost:27017/flyt");

db.vm_required_resources.insertOne({
    vm_ip: "10.129.26.124",
    host_ip: "10.129.27.234",
    compute_units: 4,
    memory: 160
});

print("Inserted document into vm_required_resources");

