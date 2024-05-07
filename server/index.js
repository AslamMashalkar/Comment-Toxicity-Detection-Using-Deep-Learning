import express from "express";
import bodyParser from "body-parser";
import mongoose from "mongoose";
import dotenv from "dotenv";
import AuthRoute from "./Routes/AuthRoute.js"
import UserRoute from "./Routes/UserRoute.js"
import PostRoute from "./Routes/PostRoute.js"
import cors from "cors"

//Routers


const app = express();
app.use(bodyParser.json({limit:'30mb', extended:true}));
app.use(bodyParser.urlencoded({limit:'30mb', extended:true}));

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', 'http://localhost:3000'); // Replace this with the appropriate origin
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});
dotenv.config()

mongoose
.connect(process.env.MONGO_DB, 
{useNewUrlParser:true, useUnifiedTopology:true}
)
.then(()=> app.listen(process.env.PORT, ()=> console.log(`Listening at ${process.env.PORT}`))).catch((error)=> console.log(error));

//usage of routes
app.use('/auth', AuthRoute)
app.use('/user', UserRoute)
app.use('/post', PostRoute)