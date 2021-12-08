const express = require("express");
const app = express();
const fs = require("fs");
const mongodb = require('mongodb');
const url = 'mongodb://localhost:27017';

app.get("/", function (req, res) {
  res.sendFile(__dirname + "/index.html");
});

// Sorry about this monstrosity -- just for demo purposes
app.get('/init-video', function (req, res) {
  mongodb.MongoClient.connect(url, function (error, client) {
    if (error) {
      res.json(error);
      return;
    }
    // connect to the videos database
    const db = client.db('videos');

    // Create GridFS bucket to upload a large file
    const bucket = new mongodb.GridFSBucket(db);

    // create upload stream using GridFS bucket
    const videoUploadStream = bucket.openUploadStream('bigbuck');

    // You can put your file instead of bigbuck.mp4
    const videoReadStream = fs.createReadStream('./bigbuck.mp4');

    // Finally Upload!
    videoReadStream.pipe(videoUploadStream);

    // All done!
    res.status(200).send("Done...");
  });
});

app.listen(8000, function () {
  console.log("Listening on port 8000!");
});
