# TSS
TorchServe server to deploy pytorch models

## Current models that are deployed in this project:

- **Trash detection** (Object detection)

## Project structure

| File/Folder      | Description |
| ----------- | ----------- |
| trash_detection      | the info of the trash detection model (mainly handler)       |
| docker   | docker files for dev and prod        |
| logs   | contains torchserve logs        |
| DockerfileHeroku   | docker to deploy server in heroku (not used)        |
| full_requirements.txt   | python requirements        |
| instal_req   | script to install python environment dependencies        |
| prepare   | script to create .mar files in modelstore that are used for deployment        |
| prepare_prod   | script to download the model from the container then prepare it  |
| run_dev   | script to run in development mode        |
| run_prod   | script to run in production mode        |
| run_heroku   | script to run in heroku (not used)    |
| stop   | script to stop all the working models    |

## How to run:
### normal mode:

1. Create a model store and model to download directory

`mkdir -p model_store`

2. Create python environment and install dependecies

`virtualenv env`

`source env/bin/activate`

`./install_req`

3. Prepare the .mar files

`./prepare`

4. Run the server

`./run_dev` (or `./run_prod`)

### Docker mode:

1. Build image

`docker build --tag torchserve:0.1.0 -f docker/torchserve/DockerfileProd` (Or DockerfileDev)

2. run image

`docker run -p 80:80 torchserve:0.1.0` (Expose the port you are using depending on dev and prod)

